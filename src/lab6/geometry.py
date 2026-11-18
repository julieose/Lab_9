"""Модель ЛР 6–8. Атрибуты OBJ принадлежат углу грани, а не позиции."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product

import numpy as np

from .transforms import identity, transform_points


def unit(v):
    v = np.asarray(v, float)
    lengths = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, lengths, out=np.zeros_like(v), where=lengths > 1e-14)


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float

    def as_array(self):
        return np.array([self.x, self.y, self.z])


@dataclass(frozen=True)
class VertexRef:
    position: int
    uv: int | None = None
    normal: int | None = None


@dataclass(frozen=True)
class Face:
    indices: tuple[int, ...]
    corners: tuple[VertexRef, ...] | None = None

    def __post_init__(self):
        if len(self.indices) < 3 or len(set(self.indices)) != len(self.indices):
            raise ValueError("Грань должна содержать не менее трёх различных индексов.")
        if self.corners is None:
            object.__setattr__(self, "corners", tuple(VertexRef(i) for i in self.indices))
        if tuple(c.position for c in self.corners) != self.indices:
            raise ValueError("Индексы углов не совпадают с индексами позиций.")

    def normal(self, vertices):
        p = vertices[list(self.indices)]
        # Суммируем векторные произведения от одной вершины. Работает и для вогнутой грани.
        return unit(np.cross(p[1:-1]-p[0], p[2:]-p[0]).sum(axis=0))


@dataclass
class Polyhedron:
    name: str
    vertices: list[Point3D]
    faces: list[Face]
    model_matrix: np.ndarray = field(default_factory=identity)
    texcoords: list[tuple[float, float]] = field(default_factory=list)
    normals: list[Point3D] = field(default_factory=list)

    def __post_init__(self):
        if not self.vertices or not self.faces or not np.isfinite(self.local_vertices).all():
            raise ValueError("Модель должна содержать конечные позиции и грани.")
        if not np.isfinite(self.texcoords).all() or not all(np.isfinite(n.as_array()).all() for n in self.normals):
            raise ValueError("UV и нормали должны быть конечными.")
        for face in self.faces:
            for i, t, n in ((c.position, c.uv, c.normal) for c in face.corners):
                if (not 0 <= i < len(self.vertices) or
                        t is not None and not 0 <= t < len(self.texcoords) or
                        n is not None and not 0 <= n < len(self.normals)):
                    raise ValueError("Индекс угла выходит за границы массива модели.")

    @property
    def local_vertices(self):
        return np.array([(v.x, v.y, v.z) for v in self.vertices], float)

    @property
    def world_vertices(self):
        return transform_points(self.local_vertices, self.model_matrix)

    @property
    def center(self):
        return self.world_vertices.mean(axis=0)

    @property
    def edges(self):
        return sorted({tuple(sorted((a,b))) for f in self.faces
                       for a,b in zip(f.indices, f.indices[1:]+f.indices[:1])})

    def compute_vertex_normals(self):
        positions = self.local_vertices
        sums = np.zeros_like(positions)
        for face in self.faces:
            sums[list(face.indices)] += face.normal(positions)
        return unit(sums)

    def apply(self, matrix):
        matrix = np.asarray(matrix, float)
        if matrix.shape != (4,4) or not np.isfinite(matrix).all():
            raise ValueError("Ожидается конечная матрица 4×4.")
        result = matrix @ self.model_matrix
        if not np.isfinite(result).all():
            raise ValueError("Координаты преобразования слишком велики.")
        self.model_matrix = result

    def reset(self):
        self.model_matrix = identity()


def _mesh(name, vertices, faces):
    positions = np.asarray(vertices, float)
    positions = positions - positions.mean(0)
    positions *= 1.65 / np.linalg.norm(positions, axis=1).max()
    outward = []
    for ids in faces:
        face = Face(tuple(ids))
        outward.append(tuple(ids) if face.normal(positions) @ positions[list(ids)].mean(0) > 0 else tuple(reversed(ids)))
    # Одна текстура целиком на каждую грань, с независимыми углами UV.
    texcoords, result = [], []
    for ids in outward:
        k = len(ids)
        if k == 4:
            uv = [(0,0), (1,0), (1,1), (0,1)]
        elif k == 3:
            uv = [(0,0), (1,0), (.5,1)]
        else:
            uv = [(.5+.5*np.cos(2*np.pi*i/k), .5+.5*np.sin(2*np.pi*i/k)) for i in range(k)]
        start = len(texcoords)
        texcoords.extend(uv)
        result.append(Face(ids, tuple(VertexRef(a, start+j) for j,a in enumerate(ids))))
    return Polyhedron(name, [Point3D(*xyz) for xyz in positions], result, texcoords=texcoords)


def tetrahedron():
    return _mesh("Тетраэдр", [(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)],
                 [(0,1,2),(0,3,1),(0,2,3),(1,3,2)])


def hexahedron():
    return _mesh("Гексаэдр (куб)", [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                                  (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)],
                 [(0,3,2,1),(4,5,6,7),(0,1,5,4),(3,7,6,2),(0,4,7,3),(1,2,6,5)])


def octahedron():
    return _mesh("Октаэдр", [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)],
                 [(0,2,4),(2,1,4),(1,3,4),(3,0,4),(2,0,5),(1,2,5),(3,1,5),(0,3,5)])


def _ico():
    p = (1+np.sqrt(5))/2
    xy = list(product((-1,1), repeat=2))
    vertices = np.array([(0,a,b*p) for a,b in xy]+[(a,b*p,0) for a,b in xy]+[(a*p,0,b) for a,b in xy])
    edges = np.isclose(np.linalg.norm(vertices[None,:,:]-vertices[:,None,:],axis=2), 2)
    faces = [(a,b,c) for a,b,c in combinations(range(12),3) if edges[a,b] and edges[b,c] and edges[a,c]]
    return vertices, faces


def icosahedron():
    return _mesh("Икосаэдр", *_ico())


def dodecahedron():
    vertices, triangles = _ico()
    dual = vertices[np.array(triangles)].mean(axis=1)
    faces = []
    for i, v in enumerate(vertices):
        outward = unit(v)
        right = unit(np.cross(outward, [0,0,1]))
        up = np.cross(outward, right)
        nearby = [j for j,t in enumerate(triangles) if i in t]
        faces.append(sorted(nearby,key=lambda j: np.arctan2(dual[j] @ up, dual[j] @ right)))
    return _mesh("Додекаэдр", dual, faces)


SOLIDS = {"Тетраэдр": tetrahedron, "Гексаэдр (куб)": hexahedron,
          "Октаэдр": octahedron, "Икосаэдр": icosahedron, "Додекаэдр": dodecahedron}


def save_obj(polyhedron, path):
    from .obj_io import save_obj as export
    export(polyhedron, path)
