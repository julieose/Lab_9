"""Видовая и проекционные матрицы из ЛР 6."""
import math
import numpy as np


def look_at(eye, target, up=None):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    if up is None:
        up = [0, 0, 1] if abs(forward[1]) > .985 else [0, 1, 0]
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    matrix = np.eye(4)
    matrix[:3, :3] = [right, np.cross(right, forward), -forward]
    matrix[:3, 3] = -matrix[:3, :3] @ eye
    return matrix


def perspective(fov_deg, aspect, near=.1, far=100.):
    if not (0 < fov_deg < 180 and aspect > 0 and 0 < near < far):
        raise ValueError("Некорректные параметры перспективы.")
    f = 1 / math.tan(math.radians(fov_deg) / 2)
    matrix = np.zeros((4, 4))
    matrix[0, 0], matrix[1, 1] = f / aspect, f
    matrix[2, 2], matrix[2, 3] = -(far + near)/(far - near), -2*far*near/(far - near)
    matrix[3, 2] = -1
    return matrix


def orthographic(half_height, aspect, near=.1, far=100.):
    if not (half_height > 0 and aspect > 0 and 0 < near < far):
        raise ValueError("Некорректные параметры ортографической проекции.")
    matrix = np.eye(4)
    matrix[0, 0], matrix[1, 1] = 1/(half_height*aspect), 1/half_height
    matrix[2, 2], matrix[2, 3] = -2/(far-near), -(far+near)/(far-near)
    return matrix
