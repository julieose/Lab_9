"""Ламберт на вершинах и toon по нормали фрагмента."""
import numpy as np
from .geometry import unit


def lambert(normals, points, light_position):
    # N и L должны иметь единичную длину и быть в одной системе координат.
    direction = unit(np.asarray(light_position)-points)
    return np.clip(np.sum(unit(normals)*direction, axis=-1), 0., 1.)


def gouraud_colors(normals, points, light_position, color):
    return lambert(normals, points, light_position)[...,None]*np.asarray(color, float)


def toon_colors(normals, points, light_position, color, levels=4):
    diffuse = lambert(normals, points, light_position)
    bands = np.minimum((diffuse * levels).astype(int), levels-1) / (levels-1)
    # Самая тёмная ступень видна, но остаётся частью toon-палитры.
    return (.18+.82*bands[...,None])*np.asarray(color, float)
