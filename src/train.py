import os
import math
import random
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from einops import rearrange, repeat
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

class VideoFrameDataset(Dataset):
    def __init__(self, root, n_source=1, n_target=4, size=512):
        self.root = root
        self.n_source = n_source
        self.n_target = n_target
        self.size = size
        self.videos = sorted(os.listdir(root))
        self.tfm = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def _sample_frames(self, frames):
        total = len(frames)
        idx = random.randint(0, total - 1)
        sources = [idx]
        targets = []
        while len(targets) < self.n_target:
            j = random.randint(0, total - 1)
            if j != idx:
                targets.append(j)
        return sources, targets

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i):
        vid_dir = os.path.join(self.root, self.videos[i])
        frame_paths = sorted(glob.glob(os.path.join(vid_dir, '*.jpg')))
        source_idx, target_idx = self._sample_frames(frame_paths)
        frames_src = [self.tfm(read_image(frame_paths[k]).float() / 255.) for k in source_idx]
        frames_tgt = [self.tfm(read_image(frame_paths[k]).float() / 255.) for k in target_idx]
        meta_path = os.path.join(vid_dir, 'smplx.json')
        meta = json.load(open(meta_path))
        shape = torch.tensor(meta['betas']).float()
        pose  = torch.tensor(meta['thetas']).float()
        expr  = torch.tensor(meta['expressions']).float()
        cam   = torch.tensor(meta['camera']).float()
        return torch.stack(frames_src), torch.stack(frames_tgt), shape, pose, expr, cam


def fourier_emb(x, num_bands=64):
    bands = 2. ** torch.arange(num_bands, device=x.device)
    x = x[..., None] * bands
    sin, cos = x.sin(), x.cos()
    return torch.cat([sin, cos], -1)


class MLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h * 4),
            nn.GELU(),
            nn.Linear(h * 4, d)
        )
    def forward(self, x): return self.net(x)

class Attention(nn.Module):
    def __init__(self, d, heads=8):
        super().__init__()
        self.h = heads
        self.to_qkv = nn.Linear(d, d * 3, bias=False)
        self.proj = nn.Linear(d, d)
    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        d = q.shape[-1] // self.h
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.h)
        w = q @ k.transpose(-2, -1) / math.sqrt(d)
        w = w.softmax(-1)
        out = w @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, d, heads=8):
        super().__init__()
        self.attn = Attention(d, heads)
        self.mlp  = MLP(d, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class BHTransformer(nn.Module):
    def __init__(self, d=1024, layers=12):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d) for _ in range(layers)])
    def forward(self, tokens):
        for blk in self.blocks: tokens = blk(tokens)
        return tokens


class GaussianDecoder(nn.Module):
    def __init__(self, d, n_pts=10475):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 16)
        )
        self.n_pts = n_pts
    def forward(self, x):
        x = self.mlp(x)
        return x.view(-1, self.n_pts, 16)


class LHM(nn.Module):
    def __init__(self, point_dim=3, img_token_dim=512, head_token_dim=512, num_layers=12):
        super().__init__()
        self.body_tok_proj = nn.Linear(point_dim + 256, 1024)
        self.img_tok_proj  = nn.Linear(img_token_dim, 1024)
        self.head_tok_proj = nn.Linear(head_token_dim, 1024)
        self.bht = BHTransformer(1024, num_layers)
        self.dec = GaussianDecoder(1024)
    def forward(self, body_pts, image_tok, head_tok):
        p_embed = fourier_emb(body_pts)
        body_tok = self.body_tok_proj(torch.cat([body_pts, p_embed], -1))
        img_tok  = self.img_tok_proj(image_tok)
        head_tok = self.head_tok_proj(head_tok)
        tokens   = torch.cat([body_tok, img_tok, head_tok], 1)
        fused    = self.bht(tokens)
        gauss    = self.dec(fused[:, :body_tok.shape[1]])
        return gauss


def splat_render(gaussians, pose, cam):
    return torch.zeros(gaussians.shape[0], 3, 512, 512, device=gaussians.device)

def photometric_loss(img_pred, img_gt, mask):
    return (F.l1_loss(img_pred * mask, img_gt * mask) + F.mse_loss(img_pred * mask, img_gt * mask)) * 0.5

def asap_loss(gauss_canon):
    return gauss_canon[..., :3].pow(2).mean()

def acap_loss(gauss_canon, thresh=0.0525):
    pos = gauss_canon[..., :3]
    diff = pos[:, 1:] - pos[:, :-1]
    return F.relu(diff.norm(dim=-1) - thresh).mean()

def total_loss(img_pred, img_gt, mask, gauss_canon):
    return photometric_loss(img_pred, img_gt, mask) + 50 * asap_loss(gauss_canon) + 10 * acap_loss(gauss_canon)


class LHMSystem(pl.LightningModule):
    def __init__(self, lr=4e-4, d=1024, layers=12, batch=16):
        super().__init__()
        self.save_hyperparameters()
        self.model = LHM(num_layers=layers)
        self.lr = lr
        self.batch = batch

    def forward(self, body_pts, img_tok, head_tok):
        return self.model(body_pts, img_tok, head_tok)

    def training_step(self, batch, _):
        src, tgt, shape, pose, expr, cam = batch
        B, C, H, W = src.shape
        body_pts = torch.randn(B, 10475, 3, device=self.device)
        img_tok  = torch.randn(B, 256, 512, device=self.device)
        head_tok = torch.randn(B, 32, 512, device=self.device)
        gauss = self(body_pts, img_tok, head_tok)
        out = splat_render(gauss, pose, cam)
        mask = torch.ones(B, 1, H, W, device=self.device)
        loss = total_loss(out, tgt[:,0], mask, gauss)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40_000)
        return [opt], [sched]


def main():
    datos = VideoFrameDataset('/path/to/videos', n_source=1, n_target=4, size=512)
    loader = DataLoader(datos,
                        batch_size=16,
                        shuffle=True,
                        num_workers=8,
                        persistent_workers=True,
                        pin_memory=True)
    sistema = LHMSystem(lr=4e-4, layers=12, batch=16)
    strat = DDPStrategy(find_unused_parameters=False)
    trainer = pl.Trainer(
        devices=32,
        accelerator='gpu',
        strategy=strat,
        precision='bf16-mixed',
        max_steps=40_000,
        gradient_clip_val=0.1,
        log_every_n_steps=50,
        accumulate_grad_batches=1
    )
    trainer.fit(sistema, loader)

if __name__ == '__main__':
    main()
