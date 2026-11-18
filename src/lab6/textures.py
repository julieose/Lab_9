"""UV-координаты граней, процедурная текстура и билинейный сэмплер."""
import numpy as np


def face_uv(points, normal):
    """Планарная развёртка для граней без полной UV-разметки в OBJ."""
    values = np.delete(points, int(np.argmax(np.abs(normal))), axis=1)
    lo, hi = values.min(axis=0), values.max(axis=0)
    return (values-lo)/np.maximum(hi-lo,1e-12)


def uv_grid(size=256):
    y, x = np.indices((size,size))
    u, v = x/size, 1-y/size
    checker = (np.floor(u*8)+np.floor(v*8))%2
    texture = np.where(checker[...,None]>0, [48,114,156], [192,224,240])
    marks = (x%(size//8)<2) | (y%(size//8)<2)
    texture[marks] = (19,46,72)
    # Цветовые метки показывают, где начало UV и куда растут оси.
    texture[(u<.095)&(v<.20)] = (244,122,60)
    texture[(u>.87)&(v<.20)] = (245,208,78)
    texture[(u<.12)&(v>.80)] = (94,217,165)
    return np.ascontiguousarray(texture, dtype=np.uint8)


def sample_bilinear(texture, uv):
    height,width = texture.shape[:2]
    coord = np.clip(uv,0.,1.)
    x,y = coord[:,0]*(width-1), (1-coord[:,1])*(height-1)
    x0,y0 = x.astype(int),y.astype(int)
    x1,y1 = np.minimum(x0+1,width-1),np.minimum(y0+1,height-1)
    s,t = (x-x0)[:,None],(y-y0)[:,None]
    return ((1-t)*((1-s)*texture[y0,x0]+s*texture[y0,x1])
            +t*((1-s)*texture[y1,x0]+s*texture[y1,x1]))
