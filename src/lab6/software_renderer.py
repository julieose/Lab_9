"""Собирает сцену, normal matrix, Z-буфер и режимы закраски. Без Qt."""
from dataclasses import dataclass
import numpy as np

from .geometry import unit
from .visibility import normal_matrix, is_front_facing
from .interpolation import quad_frame, quad_weights, triangulate
from .lighting import gouraud_colors, toon_colors
from .textures import face_uv, sample_bilinear, uv_grid
from .rasterizer import clip_polygon, clip_segment, to_screen, draw_triangle, draw_depth_line


DEFAULT_TEXTURE = uv_grid()


@dataclass
class RenderOptions:
    cull_backfaces: bool = True
    use_zbuffer: bool = True
    draw_faces: bool = True
    draw_edges: bool = False
    draw_normals: bool = False
    mode: str = "gouraud"
    light: tuple[float,float,float] = (3.,4.,5.)
    levels: int = 4
    texture: np.ndarray | None = None


@dataclass
class RenderResult:
    pixels: np.ndarray
    zbuffer: np.ndarray
    total_faces: int
    culled_faces: int
    rendered_faces: int


def render_scene(objects, camera, width, height, options=None):
    opts = options or RenderOptions()
    texture = opts.texture
    if opts.mode=="texture" and texture is None:
        texture = DEFAULT_TEXTURE
    vp = camera.projection_matrix(width,height) @ camera.view_matrix()
    pixels = np.zeros((height,width,4),np.uint8)
    depth = np.full((height,width),np.inf,np.float32)
    total = culled = drawn = 0
    lines = []
    for obj in objects:
        mesh = obj.model
        local,world = mesh.local_vertices,mesh.world_vertices
        clip = np.column_stack((world,np.ones(len(world)))) @ vp.T
        nm = normal_matrix(mesh.model_matrix[:3,:3])
        local_normals = np.array([face.normal(local) for face in mesh.faces])
        face_normals = unit(local_normals @ nm.T)
        smooth = np.zeros_like(local)
        for face,n in zip(mesh.faces,local_normals):
            smooth[list(face.indices)] += n
        smooth = unit(unit(smooth) @ nm.T)
        imported = unit(np.array([n.as_array() for n in mesh.normals]) @ nm.T) if mesh.normals else None
        edges = set()
        for fi, face in enumerate(mesh.faces):
            total += 1
            ids = list(face.indices)
            positions = world[ids]
            normal = face_normals[fi]
            center = positions.mean(0)
            if opts.cull_backfaces and not is_front_facing(normal,center,camera):
                culled += 1
                continue
            if opts.draw_edges:
                edges.update(tuple(sorted((a,b))) for a,b in zip(ids,ids[1:]+ids[:1]))
            if opts.draw_normals:
                length = max(.06,np.linalg.norm(positions-center,axis=1).max()*.55)
                ends = np.column_stack(([center,center+length*normal],[1.,1.])) @ vp.T
                lines.append((*ends,(254,204,93),False))
            if not opts.draw_faces:
                continue
            color = obj.color
            polygons = triangulate(local[ids],local_normals[fi])
            face_drawn = False
            if opts.mode in ("gouraud","toon"):
                normals = np.array([
                    imported[c.normal] if c.normal is not None and np.any(imported[c.normal]) else smooth[c.position]
                    for c in face.corners])
                normals[~np.any(normals,axis=1)] = normal
            if opts.mode == "gouraud":
                colors = gouraud_colors(normals,positions,opts.light,color)
                shade = lambda weights: weights@colors
            elif opts.mode == "toon":
                shade = lambda weights: toon_colors(weights@normals,weights@positions,opts.light,color,opts.levels)
            elif opts.mode == "texture":
                uv = (np.array([mesh.texcoords[c.uv] for c in face.corners])
                      if all(c.uv is not None for c in face.corners) else face_uv(local[ids],local_normals[fi]))
                shade = lambda weights: sample_bilinear(texture,weights@uv)
            else:
                shade = None
                color = tuple(round(c*(.72+.07*(fi%5))) for c in color)
            frame = quad_frame(local[ids],local_normals[fi]) if shade is not None else None
            identity = np.eye(len(ids)) if shade is not None else None
            for triangle in polygons:
                corners = list(triangle)
                positions_clip = clip[np.array(ids)[corners]]
                if shade is not None:
                    positions_clip = np.column_stack((positions_clip,identity[corners]))
                polygon = clip_polygon(positions_clip)
                if len(polygon)<3 or np.any(polygon[:,3]<=0):
                    continue
                face_drawn = True
                screen = to_screen(polygon,width,height)
                inv_w = 1/polygon[:,3]
                shader = shade
                if frame is not None:
                    shader = lambda weights: shade(quad_weights(frame,weights))
                for i in range(1,len(polygon)-1):
                    t = [0,i,i+1]
                    draw_triangle(screen[t],inv_w[t],color,pixels,depth,opts.use_zbuffer,
                                  basis=polygon[t,4:],shader=shader)
            drawn += face_drawn
        if opts.draw_edges:
            lines.extend((clip[a],clip[b],(203,225,249),opts.draw_faces and opts.use_zbuffer) for a,b in edges)
    # Сначала заполняем глубину всех моделей, потом накладываем рёбра.
    for a,b,color,test_depth in lines:
        seg = clip_segment(a,b)
        if len(seg)==2 and np.all(seg[:,3]>0):
            draw_depth_line(to_screen(seg,width,height),pixels,depth,color,test_depth)
    return RenderResult(pixels,depth,total,culled,drawn)
