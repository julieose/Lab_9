"""Камера из ЛР 8. Орбита меняет положение глаза, а не модели."""
from dataclasses import dataclass, field
import numpy as np

from .projection import look_at, orthographic, perspective


@dataclass
class Camera:
    position: np.ndarray = field(default_factory=lambda: np.array([5., 3., 6.]))
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    projection: str = "perspective"
    fov_degrees: float = 48.
    near: float = .05
    far: float = 1000.
    ortho_half_height: float = 3.

    def __post_init__(self):
        self.position = np.asarray(self.position, float)
        self.target = np.asarray(self.target, float)
        if not np.isfinite(self.position).all() or not np.isfinite(self.target).all() or np.linalg.norm(self.target-self.position) < 1e-8:
            raise ValueError("Камера и цель должны быть различными конечными точками.")

    @property
    def direction(self):
        delta = self.target - self.position
        return delta / np.linalg.norm(delta)

    def view_matrix(self):
        return look_at(self.position, self.target)

    def projection_matrix(self, width, height):
        aspect = width / max(1, height)
        if self.projection == "orthographic":
            return orthographic(self.ortho_half_height, aspect, self.near, self.far)
        return perspective(self.fov_degrees, aspect, self.near, self.far)

    def set_pose(self, position, direction):
        position, direction = np.asarray(position, float), np.asarray(direction, float)
        length = np.linalg.norm(direction)
        if not np.isfinite(position).all() or not np.isfinite(length) or length < 1e-8:
            raise ValueError("Задайте конечную позицию и ненулевой вектор обзора.")
        distance = np.linalg.norm(self.target-self.position)
        self.position = position
        self.target = position + direction / length * distance
        self.far = max(self.far, distance * 6)

    def orbit(self, yaw_degrees, pitch_degrees=0.):
        offset = self.position-self.target
        r = np.linalg.norm(offset)
        yaw = np.arctan2(offset[0], offset[2]) + np.radians(yaw_degrees)
        pitch = np.clip(np.arcsin(np.clip(offset[1]/r, -1, 1)) + np.radians(pitch_degrees),
                        np.radians(-88.), np.radians(88.))
        self.position = self.target + r * np.array([
            np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)])

    def zoom(self, factor):
        offset = self.position-self.target
        r = np.linalg.norm(offset)
        self.position = self.target + offset/r * np.clip(r*factor, self.near*2, self.far*.6)
        self.ortho_half_height *= factor

    def fit(self, points, aspect=1.):
        points = np.asarray(points, float)
        self.target = (points.min(0) + points.max(0)) / 2
        radius = max(.05, float(np.linalg.norm(points-self.target, axis=1).max()))
        gaze = np.array([2.35, 1.45, 2.85])
        vertical_tan = np.tan(np.radians(self.fov_degrees*.5))
        distance = radius * max(3.1, 1.35 / (vertical_tan * min(1., aspect)))
        self.position = self.target + distance * gaze / np.linalg.norm(gaze)
        self.ortho_half_height = radius * 1.35 / min(1., aspect)
        self.near, self.far = max(distance*.02, 1e-4), max(distance*30, 100.)
