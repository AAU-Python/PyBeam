"""Dataclasses for representing the finite element model."""
import attrs
import numpy as np


@attrs.define
class Node:
    """A node in the finite element model."""

    x: float
    y: float
    index: int
    dofs: tuple[int, int, int] = attrs.field()

    @dofs.default
    def _set_dofs(self) -> tuple[int, int, int]:
        dofs = tuple(self.index * 3 + i for i in range(3))
        return dofs


@attrs.define
class BeamElement:
    """A 2D Bernoulli-Euler beam element."""

    start_node: Node
    end_node: Node
    index: int
    modulus_of_elasticity: float
    moment_of_intertia: float
    area: float
    length: float = attrs.field()
    angle: float = attrs.field()

    @length.default
    def _set_length(self) -> float:
        dx = self.end_node.x - self.start_node.x
        dy = self.end_node.y - self.start_node.y
        return np.linalg.norm((dx, dy))

    @angle.default
    def _set_angle(self) -> float:
        dx = self.end_node.x - self.start_node.x
        angle = np.arccos(dx / self.length)
        return angle
