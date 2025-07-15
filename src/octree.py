from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import numpy as np


@dataclass
class NodoOctree:
    centro: np.ndarray
    mitad: float
    profundidad: int
    max_profundidad: int
    max_puntos: int
    hijos: Optional[List["NodoOctree"]] = field(default=None)
    puntos: Optional[List[np.ndarray]] = field(default_factory=list)

    def insertar(self, p: Sequence[float]) -> None:
        p = np.asarray(p, dtype=float)

        if self.hijos is not None:
            self._hijo_adecuado(p).insertar(p)
            return

        self.puntos.append(p)

        if len(self.puntos) > self.max_puntos and self.profundidad < self.max_profundidad:
            self._subdividir()
            for q in self.puntos:
                self._hijo_adecuado(q).insertar(q)
            self.puntos = None

    def buscar_rango(self,
                     centro: Sequence[float],
                     radio: float,
                     encontrados: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:

        if encontrados is None:
            encontrados = []

        if not self._intersecta_esfera(np.asarray(centro, dtype=float), radio):
            return encontrados

        if self.hijos is not None:
            for h in self.hijos:
                h.buscar_rango(centro, radio, encontrados)
        elif self.puntos is not None:
            for p in self.puntos:
                if np.linalg.norm(p - centro) <= radio:
                    encontrados.append(p)

        return encontrados

    def _hijo_adecuado(self, p: np.ndarray) -> "NodoOctree":
        idx = 0
        if p[0] > self.centro[0]:
            idx |= 1
        if p[1] > self.centro[1]:
            idx |= 2
        if p[2] > self.centro[2]:
            idx |= 4
        return self.hijos[idx]

    def _subdividir(self) -> None:
        cuarto = self.mitad / 2.0
        offsets = np.array([[dx, dy, dz]
                            for dx in (-cuarto, cuarto)
                            for dy in (-cuarto, cuarto)
                            for dz in (-cuarto, cuarto)])
        self.hijos = [
            NodoOctree(self.centro + off,
                       cuarto,
                       self.profundidad + 1,
                       self.max_profundidad,
                       self.max_puntos)
            for off in offsets
        ]

    def _intersecta_esfera(self, c: np.ndarray, r: float) -> bool:
        d = np.maximum(np.abs(c - self.centro) - self.mitad, 0.0)
        return (d**2).sum() <= r * r


class Octree:
    def __init__(self,
                 lado: float = 2.0,
                 max_profundidad: int = 8,
                 max_puntos: int = 16):
        mitad = lado / 2.0
        self.raiz = NodoOctree(centro=np.zeros(3),
                               mitad=mitad,
                               profundidad=0,
                               max_profundidad=max_profundidad,
                               max_puntos=max_puntos)

    def insertar(self, p: Sequence[float]) -> None:
        self.raiz.insertar(p)

    def buscar_rango(self, centro: Sequence[float], radio: float) -> List[np.ndarray]:
        return self.raiz.buscar_rango(centro, radio)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    puntos = rng.uniform(-1, 1, size=(50_000, 3))

    octree = Octree(lado=2.0, max_profundidad=8, max_puntos=32)
    for p in puntos:
        octree.insertar(p)

    cerca = octree.buscar_rango([0, 0, 0], 0.05)
    print(f"Vecinos encontrados: {len(cerca)}")
