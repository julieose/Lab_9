"""Интерполяция атрибутов и разбиение граней для программного рендера."""
import numpy as np


def cross2(a,b):
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def _planar_coordinates(points, normal):
    return np.delete(points-points[0],int(np.argmax(np.abs(normal))),axis=1)


def quad_frame(points, normal):
    """Кадр для обратного билинейного отображения выпуклой плоской грани."""
    if len(points) != 4:
        return None
    scale = np.linalg.norm(points-points[0],axis=1).max()
    if scale < 1e-12 or np.max(np.abs((points-points[0]) @ normal)) > scale*1e-9:
        return None
    q = _planar_coordinates(points,normal)/scale
    edges = np.roll(q,-1,axis=0)-q
    turns = cross2(edges,np.roll(edges,-1,axis=0))
    if not (np.all(turns>1e-10) or np.all(turns< -1e-10)):
        return None
    return q


def quad_weights(frame, weights):
    """Восстанавливаем s,t из точки на грани и считаем четыре билинейных веса."""
    p = weights @ frame
    b,c,d = frame[1],frame[3],frame[2]-frame[1]-frame[3]
    determinant = cross2(b,c)
    s,t = cross2(p,c)/determinant,cross2(b,p)/determinant
    if np.linalg.norm(d)>1e-10:
        # Решаем p = b*s + c*t + d*s*t методом Ньютона.
        s,t = weights[:,1]+weights[:,2],weights[:,2]+weights[:,3]
        for _ in range(9):
            error = s[:,None]*b+t[:,None]*c+(s*t)[:,None]*d-p
            if np.max(np.abs(error)) < 1e-11:
                break
            ds,dt = b+t[:,None]*d,c+s[:,None]*d
            jac = cross2(ds,dt)
            valid = np.abs(jac)>1e-12
            s -= np.divide(cross2(error,dt),jac,out=np.zeros_like(s),where=valid)
            t -= np.divide(cross2(ds,error),jac,out=np.zeros_like(t),where=valid)
    s,t = np.clip(s,0,1),np.clip(t,0,1)
    return np.column_stack(((1-s)*(1-t),s*(1-t),s*t,(1-s)*t))


def triangulate(points, normal):
    """Треугольники выпуклого полигона или отсечение ушей у вогнутого."""
    n = len(points)
    if n == 3:
        return [(0,1,2)]
    q = _planar_coordinates(points,normal)
    q /= max(1e-15,np.linalg.norm(q,axis=1).max())
    sign = np.sign(cross2(q,np.roll(q,-1,axis=0)).sum())
    edges = np.roll(q,-1,axis=0)-q
    if np.all(sign*cross2(edges,np.roll(edges,-1,axis=0))>=-1e-10):
        return [(0,i,i+1) for i in range(1,n-1)]
    ring = list(range(n))
    triangles = []
    while len(ring)>3:
        for i,b in enumerate(ring):
            a,c = ring[i-1],ring[(i+1)%len(ring)]
            pa,pb,pc = q[[a,b,c]]
            if sign*cross2(pb-pa,pc-pa)<=1e-10:
                continue
            rest = q[[j for j in ring if j not in (a,b,c)]]
            occupied = ((sign*cross2(pb-pa,rest-pa)>=-1e-10)
                        &(sign*cross2(pc-pb,rest-pb)>=-1e-10)
                        &(sign*cross2(pa-pc,rest-pc)>=-1e-10))
            if not occupied.any():
                triangles.append((a,b,c))
                ring.pop(i)
                break
        else:
            return []  # Вырожденный или самопересекающийся OBJ-полигон.
    return triangles+[tuple(ring)]
