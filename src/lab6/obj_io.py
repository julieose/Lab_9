"""Импорт/экспорт Wavefront OBJ с раздельными v, vt и vn."""
from pathlib import Path
import numpy as np

from .geometry import Face, Point3D, Polyhedron, VertexRef, unit


def load_obj(path):
    path = Path(path)
    positions, uvs, normals, faces = [], [], [], []
    name = path.stem
    pending = ""
    with path.open(encoding="utf-8-sig", errors="replace") as source:
        for number, line in enumerate(source, 1):
            line = line.split("#",1)[0].strip()
            if line.endswith(chr(92)):
                pending += line[:-1] + " "
                continue
            words = (pending+line).split()
            pending = ""
            if not words:
                continue
            key, args = words[0], words[1:]
            try:
                if key == "v":
                    xyz = np.array(args[:3],float)
                    if len(xyz) != 3:
                        raise ValueError("позиции нужны 3 координаты")
                    if len(args) == 4:
                        w = float(args[3])
                        if w == 0 or not np.isfinite(w):
                            raise ValueError("недопустимое w")
                        xyz /= w
                    positions.append(Point3D(*xyz))
                elif key == "vt":
                    uvs.append((float(args[0]), float(args[1]) if len(args)>1 else 0.))
                elif key == "vn":
                    normals.append(Point3D(*map(float,args[:3])))
                elif key == "o" and args and not faces:
                    name = " ".join(args)
                elif key == "f":
                    corners = []
                    for token in args:
                        fields = token.split("/")
                        if len(fields)>3 or not fields[0]:
                            raise ValueError(f"неверный угол {token}")
                        values = []
                        for field, seq in zip((fields+["",""])[:3], (positions,uvs,normals)):
                            if not field:
                                values.append(None)
                                continue
                            k = int(field)
                            index = k-1 if k>0 else len(seq)+k
                            if not 0 <= index < len(seq):
                                raise ValueError(f"индекс {k} вне диапазона")
                            values.append(index)
                        corners.append(VertexRef(*values))
                    if len(corners)>3 and corners[0] == corners[-1]:
                        corners.pop()
                    faces.append(Face(tuple(c.position for c in corners),tuple(corners)))
            except (ValueError, TypeError, IndexError) as err:
                raise ValueError(f"OBJ, строка {number}: {err}") from err
    if pending:
        raise ValueError("Незавершённая строка OBJ.")
    return Polyhedron(name, positions, faces, texcoords=uvs, normals=normals)


def save_obj(model, path):
    linear = model.model_matrix[:3,:3]
    flip = np.linalg.det(linear)<0
    normals = [n.as_array() for n in model.normals]
    start = len(normals)
    if any(c.normal is None for f in model.faces for c in f.corners):
        normals.extend(model.compute_vertex_normals())
    vals = unit(np.array(normals) @ np.linalg.pinv(linear)) if normals else []
    with Path(path).open("w",encoding="utf-8",newline="\n") as stream:
        stream.write("o "+" ".join(model.name.splitlines())+"\n")
        for key, array in (("v",model.world_vertices),("vt",model.texcoords),("vn",vals)):
            for row in array:
                stream.write(key+" "+" ".join(f"{v:.10g}" for v in row)+"\n")
        for face in model.faces:
            corners = face.corners[::-1] if flip else face.corners
            tokens = []
            for c in corners:
                t = "" if c.uv is None else str(c.uv+1)
                n = start+c.position if c.normal is None else c.normal
                tokens.append(f"{c.position+1}/{t}/{n+1}")
            stream.write("f "+" ".join(tokens)+"\n")
