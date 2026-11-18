"""Отсечение и программная растеризация из ЛР 8 с атрибутами для ЛР 9."""
import numpy as np

PLANES = np.array([[1,0,0,1],[-1,0,0,1],[0,1,0,1],[0,-1,0,1],[0,0,1,1],[0,0,-1,1]],float)


def clip_polygon(vertices):
    """Первые четыре столбца — clip-координаты, остальные — атрибуты."""
    plane_values = vertices[:,:4] @ PLANES.T
    if np.all(plane_values>=0):
        return vertices
    if np.any(np.all(plane_values<0,axis=0)):
        return vertices[:0]
    polygon = vertices
    for plane in PLANES:
        if not len(polygon):
            break
        values = polygon[:,:4] @ plane
        result = []
        for i,b in enumerate(values):
            a = values[i-1]
            if (a>=0)!=(b>=0):
                t = a/(a-b)
                result.append(polygon[i-1]+t*(polygon[i]-polygon[i-1]))
            if b>=0:
                result.append(polygon[i])
        polygon = np.array(result).reshape(-1,vertices.shape[1])
    return polygon


def clip_segment(a, b):
    va,vb = PLANES@a,PLANES@b
    low,high = 0.,1.
    for u,v in zip(va,vb):
        if u<0 and v<0:
            return np.empty((0,4))
        if u<0:
            low = max(low,u/(u-v))
        elif v<0:
            high = min(high,u/(u-v))
    if low>high:
        return np.empty((0,4))
    return np.array([a+low*(b-a),a+high*(b-a)])


def to_screen(clip,w,h):
    ndc = clip[:,:3]/clip[:,3,None]
    return np.column_stack(((ndc[:,0]+1)*w/2,(1-ndc[:,1])*h/2,ndc[:,2]))


def draw_triangle(p, inv_w, color, pixels, zbuffer, use_zbuffer=True, *, basis=None, shader=None):
    h,w = zbuffer.shape
    xl,yl = np.maximum([0,0],np.floor(p[:,:2].min(0)).astype(int))
    xr,yr = np.minimum([w-1,h-1],np.ceil(p[:,:2].max(0)).astype(int))
    if xl>xr or yl>yr:
        return
    (x0,y0),(x1,y1),(x2,y2) = p[:,:2]
    area = (x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)
    if abs(area)<1e-10:
        return
    yy,xx = np.ogrid[yl:yr+1,xl:xr+1]
    x,y = xx+.5,yy+.5
    b0 = ((x1-x)*(y2-y)-(y1-y)*(x2-x))/area
    b1 = ((x2-x)*(y0-y)-(y2-y)*(x0-x))/area
    b2 = 1-b0-b1
    z = (b0*p[0,2]+b1*p[1,2]+b2*p[2,2]).astype(zbuffer.dtype)
    tile = zbuffer[yl:yr+1,xl:xr+1]
    mask = (b0>=-1e-8)&(b1>=-1e-8)&(b2>=-1e-8)
    if use_zbuffer:
        mask &= z<tile
    if not mask.any():
        return
    tile[mask] = z[mask]
    output = pixels[yl:yr+1,xl:xr+1]
    if shader is not None:
        # Для атрибутов, в отличие от z_NDC, нужна коррекция по 1/w.
        weights = np.column_stack((b0[mask],b1[mask],b2[mask]))*inv_w
        weights /= weights.sum(axis=1,keepdims=True)
        color = np.rint(shader(weights@basis)).clip(0,255).astype(np.uint8)
    output[mask,:3] = color
    output[mask,3] = 255


def draw_depth_line(p, pixels, zbuffer, color, depth_test):
    h,w = zbuffer.shape
    samples = max(2,int(np.abs(p[1,:2]-p[0,:2]).max())+1)
    a = np.linspace(0,1,samples)[:,None]
    pts = p[0]+a*(p[1]-p[0])
    x,y = np.floor(pts[:,:2]).astype(int).T
    mask = (0<=x)&(x<w)&(0<=y)&(y<h)
    x,y,z = x[mask],y[mask],pts[mask,2]
    if depth_test:
        visible = z<=zbuffer[y,x]+.0005
        x,y = x[visible],y[visible]
    pixels[y,x] = (*color,255)
