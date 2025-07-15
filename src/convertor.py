import argparse
import pathlib
import sys
import trimesh


def load_glb(path: pathlib.Path) -> trimesh.Trimesh:
    """Carga un archivo GLB y devuelve una sola malla fusionada."""
    scene_or_mesh = trimesh.load(path, force='scene')
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh

    meshes = [g for g in scene_or_mesh.geometry.values()]
    if not meshes:
        raise ValueError("No se encontraron mallas en el GLB")
    merged = trimesh.util.concatenate(meshes)
    return merged


def export_mesh(mesh: trimesh.Trimesh, dest: pathlib.Path, fmt: str):
    """Exporta la malla al formato indicado."""
    fmt = fmt.lower()
    if fmt not in {"obj", "ply", "stl"}:
        raise ValueError(f"Formato no soportado: {fmt}")

    data = None
    if fmt == "ply":
        data = mesh.export(file_type="ply")
    elif fmt == "obj":
        data = mesh.export(file_type="obj")
    elif fmt == "stl":
        data = mesh.export(file_type="stl")

    dest.write_bytes(data)
    print(f"✓ Guardado: {dest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convierte un archivo .glb a obj, ply y/o stl")
    parser.add_argument("glb_file", type=pathlib.Path, help="Archivo .glb de entrada")
    parser.add_argument(
        "--to", nargs="+", default=["obj", "ply"],
        help="Formatos de salida (obj, ply, stl). Por defecto: obj ply")
    parser.add_argument(
        "-o", "--outdir", type=pathlib.Path, default=None,
        help="Carpeta donde guardar los archivos (por defecto junto al GLB)")

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.glb_file.exists():
        sys.exit(f"Archivo no encontrado: {args.glb_file}")

    outdir = args.outdir or args.glb_file.parent
    outdir.mkdir(parents=True, exist_ok=True)

    print("Cargando GLB...")
    mesh = load_glb(args.glb_file)

    for fmt in args.to:
        dest = outdir / f"{args.glb_file.stem}.{fmt.lower()}"
        export_mesh(mesh, dest, fmt)

    print("Conversión terminada.")


if __name__ == "__main__":
    main()
