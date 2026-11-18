"""Фигуры вращения и сетки функций из ЛР 7."""
import ast
import numpy as np

from .geometry import Face, Point3D, Polyhedron, VertexRef
from .transforms import rotation_axis, transform_points

PROFILE_DEFAULT = """0 -1.4 0
.65 -1.4 0
.80 -1.1 0
.55 -.3 0
.35 .15 0
.55 .9 0
.55 1.25 0
0 1.25 0"""
FUNCTIONS = {"Волны": "sin(x)*cos(y)", "Круговая волна": "sin(sqrt(x*x+y*y))",
             "Седло": "0.15*(x*x-y*y)", "Параболоид": "0.12*(x*x+y*y)"}


def read_profile(text):
    points = []
    for i, line in enumerate(text.splitlines(), 1):
        values = line.split("#", 1)[0].replace(";", " ").replace(",", " ").split()
        if not values:
            continue
        try:
            xyz = np.array(values, float)
            if len(xyz) != 3 or not np.isfinite(xyz).all():
                raise ValueError
        except ValueError:
            raise ValueError(f"Строка {i}: нужны три конечных числа X Y Z.") from None
        points.append(Point3D(*xyz))
    if not 2 <= len(points) <= 250:
        raise ValueError("Введите от 2 до 250 точек.")
    return points


def revolution(profile, axis, steps):
    if axis not in "XYZ" or len(axis) != 1 or not 3 <= steps <= 256 or len(profile) < 2:
        raise ValueError("Выберите ось X/Y/Z, задайте 3–256 секторов и не менее двух точек.")
    base = np.array([p.as_array() for p in profile])
    a = "XYZ".index(axis)
    radial = np.delete(base, a, axis=1)
    poles = np.linalg.norm(radial, axis=1) < 1e-9
    base[np.ix_(poles, [i for i in range(3) if i != a])] = 0
    n = len(base)
    vertices, rings, faces = [], [], []
    uv = [(j/steps, k/(n-1)) for j in range(steps+1) for k in range(n)]
    for j in range(steps):
        ids = []
        for k, p in enumerate(transform_points(base, rotation_axis(axis, j*360/steps))):
            if j and poles[k]:
                ids.append(rings[0][k])
            else:
                ids.append(len(vertices))
                vertices.append(p)
        rings.append(ids)
    for j in range(steps):
        r, s = rings[j], rings[(j+1)%steps]
        for k in range(n-1):
            if np.linalg.norm(base[k+1]-base[k]) < 1e-9:
                continue
            raw = [(r[k],j*n+k), (s[k],(j+1)*n+k),
                   (s[k+1],(j+1)*n+k+1), (r[k+1],j*n+k+1)]
            pairs = list({p: t for p,t in raw}.items())
            if len(pairs) >= 3:
                faces.append(pairs)
    if not faces:
        raise ValueError("Не получилось построить ни одной грани.")
    points = np.array(vertices)
    # Для закрытой сетки знак объёма задаёт внешний обход даже у вогнутой фигуры.
    sign = base[-1,a]-base[0,a]
    if poles[0] and poles[-1]:
        center = points.mean(0)
        volume = 0.
        for pairs in faces:
            q = points[[i for i,_ in pairs]]-center
            volume += np.cross(q[1:-1],q[2:]).sum(axis=0) @ q[0]
        if abs(volume) > 1e-12:
            sign = volume
    result = []
    for pairs in faces:
        corners = tuple(VertexRef(i,t) for i,t in (pairs[::-1] if sign < 0 else pairs))
        result.append(Face(tuple(c.position for c in corners), corners))
    return Polyhedron(f"Фигура вращения ({axis}, {steps})", [Point3D(*v) for v in points], result, texcoords=uv)


FUN = {key: getattr(np, key) for key in ("sin","cos","tan","sqrt","abs","exp","log")}
OPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide,
       ast.Pow: np.power, ast.Mod: np.remainder}


def _evaluate(node, x, y):
    if isinstance(node, ast.Constant) and type(node.value) in (float,int) and abs(node.value) < 1e9:
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in ("x","y","pi"):
        return {"x":x,"y":y,"pi":np.pi}[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_evaluate(node.left,x,y),_evaluate(node.right,x,y))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd,ast.USub)):
        return (-1 if isinstance(node.op,ast.USub) else 1)*_evaluate(node.operand,x,y)
    if (isinstance(node, ast.Call) and isinstance(node.func,ast.Name)
            and node.func.id in FUN and len(node.args)==1 and not node.keywords):
        return FUN[node.func.id](_evaluate(node.args[0],x,y))
    raise ValueError("Разрешены x, y, pi, числа, + − * / ** % и sin, cos, tan, sqrt, abs, exp, log.")


def function_surface(expr, x0, x1, y0, y1, nx, ny):
    if len(expr) > 180 or not (x0 < x1 and y0 < y1 and 2 <= nx <= 150 and 2 <= ny <= 150):
        raise ValueError("Проверьте формулу, интервалы и разбиения от 2 до 150.")
    xx, yy = np.meshgrid(np.linspace(x0,x1,nx+1), np.linspace(y0,y1,ny+1))
    try:
        with np.errstate(all="ignore"):
            zz = np.broadcast_to(_evaluate(ast.parse(expr.strip(),mode="eval").body,xx,yy), xx.shape)
    except SyntaxError:
        raise ValueError("Синтаксическая ошибка в формуле.") from None
    if not np.isfinite(zz).all() or np.abs(zz).max() > 1e6:
        raise ValueError("Функция не задаёт конечную поверхность на этом диапазоне.")
    pts = [Point3D(*v) for v in zip(xx.flat, yy.flat, zz.flat)]
    uv = [(i/nx,j/ny) for j in range(ny+1) for i in range(nx+1)]
    faces = []
    for j in range(ny):
        for i in range(nx):
            a = j*(nx+1)+i
            ids = (a,a+1,a+nx+2,a+nx+1)
            faces.append(Face(ids,tuple(VertexRef(k,k) for k in ids)))
    return Polyhedron(f"z = {expr}", pts, faces, texcoords=uv)


def sphere():
    angles = np.linspace(-np.pi/2, np.pi/2, 15)
    mesh = revolution([Point3D(1.3*np.cos(a), 1.3*np.sin(a), 0.) for a in angles], "Y", 28)
    positions = mesh.local_vertices
    mesh.normals = [Point3D(*(p/np.linalg.norm(p))) for p in positions]
    mesh.faces = [Face(f.indices, tuple(VertexRef(c.position,c.uv,c.position) for c in f.corners)) for f in mesh.faces]
    mesh.name = "Сфера"
    return mesh
