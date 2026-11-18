"""Модельные матрицы 4×4 для однородных столбцов."""
from __future__ import annotations

import math
import numpy as np


def identity():
    return np.eye(4)


def translation(dx, dy, dz):
    matrix = identity()
    matrix[:3, 3] = dx, dy, dz
    return matrix


def scaling(sx, sy, sz):
    return np.diag([sx, sy, sz, 1.])


def _axis_rotation(i, angle):
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    j, k = (i + 1) % 3, (i + 2) % 3
    matrix = identity()
    matrix[j, j] = matrix[k, k] = c
    matrix[j, k], matrix[k, j] = -s, s
    return matrix


def rotation_x(angle):
    return _axis_rotation(0, angle)


def rotation_y(angle):
    return _axis_rotation(1, angle)


def rotation_z(angle):
    return _axis_rotation(2, angle)


def rotation_axis(axis, angle):
    return {"X": rotation_x, "Y": rotation_y, "Z": rotation_z}[axis.upper()](angle)


def reflection(plane):
    return scaling(*{"XY": (1, 1, -1), "XZ": (1, -1, 1), "YZ": (-1, 1, 1)}[plane.upper()])


def around_point(matrix, point):
    point = np.asarray(point, float)
    return translation(*point) @ matrix @ translation(*-point)


def rotate_around_line(a, b, angle):
    a, b = np.asarray(a, float), np.asarray(b, float)
    direction = b - a
    length = np.linalg.norm(direction)
    if not np.isfinite(length) or length < 1e-10:
        raise ValueError("Точки A и B должны отличаться.")
    l, m, n = direction / length
    psi = math.degrees(math.atan2(m, n))
    theta = math.degrees(math.atan2(l, math.hypot(m, n)))
    align = rotation_y(-theta) @ rotation_x(psi)
    return around_point(align.T @ rotation_z(angle) @ align, a)


def transform_points(points, matrix):
    points = np.asarray(points, float)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    result = homogeneous @ np.asarray(matrix, float).T
    if np.any(np.abs(result[:, 3]) < 1e-12):
        raise ValueError("Однородная координата w равна нулю.")
    return result[:, :3] / result[:, 3, None]
