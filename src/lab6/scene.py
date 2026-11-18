"""Сцена ЛР 8, трансформируемые модели и цвета для демонстрации перекрытий."""
from dataclasses import dataclass
from .geometry import Polyhedron, hexahedron, tetrahedron, octahedron
from .transforms import translation, scaling


@dataclass
class SceneObject:
    model: Polyhedron
    color: tuple[int,int,int] = (71,170,237)
    label: str = ""

    @property
    def name(self):
        return self.label or self.model.name


def overlap_scene():
    items = []
    for model, xyz, color in [
        (hexahedron(), (.2,0,.5), (71,160,239)),
        (tetrahedron(), (.6,.1,-.8), (241,124,116)),
        (octahedron(), (-.4,-.1,-1), (84,209,170)),
    ]:
        model.apply(translation(*xyz) @ scaling(.9,.9,.9))
        items.append(SceneObject(model,color))
    return items
