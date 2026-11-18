"""Нормали и тест ориентации грани из ЛР 8."""
import numpy as np
from .geometry import unit


def normal_matrix(linear):
    return np.linalg.pinv(linear).T


def face_normals(model):
    points = model.local_vertices
    return unit(np.array([f.normal(points) for f in model.faces]) @ normal_matrix(model.model_matrix[:3,:3]).T)


def is_front_facing(normal, center, camera):
    to_eye = -camera.direction if camera.projection == "orthographic" else camera.position-center
    return float(normal @ to_eye) > 1e-10
