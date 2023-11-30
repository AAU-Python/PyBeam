"""Dataclasses for representing the finite element model."""
import csv
import logging

import attrs
import numpy as np

_LOGGER = logging.getLogger(__name__)


@attrs.define(frozen=True)
class Node:
    """A node in the finite element model."""

    x: float = attrs.field(converter=float)
    y: float = attrs.field(converter=float)
    index: int = attrs.field(converter=int)
    dofs: tuple[int, int, int] = attrs.field(init=False)

    @dofs.default
    def _set_dofs(self) -> tuple[int, int, int]:
        dofs = tuple(self.index * 3 + i for i in range(3))
        return dofs


def nodes_from_csv(file_path: str) -> list[Node]:
    """Serialize ``Node`` instances from a CSV file."""
    nodes: list[Node] = []
    existing_indices = {}
    with open(file_path) as file_buffer:
        reader = csv.DictReader(file_buffer)
        for record in reader:
            if record["index"] in existing_indices:
                _LOGGER.warning(f"Duplicate node index: {record["index"]}")
            nodes.append(Node(**record))
    return nodes


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
