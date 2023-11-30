"""Dataclasses for representing the finite element model."""
import csv
import logging

import attrs
import numpy as np

_LOGGER = logging.getLogger(__name__)

def _parse_bool_from_str(string: str) -> bool:
    """Parse a boolean from a string.

    Example:
        >>> _parse_bool_from_str("true")
        True
    """
    return string == "true"

@attrs.define(frozen=True)
class Node:
    """A node in the finite element model."""

    x: float = attrs.field(converter=float)
    y: float = attrs.field(converter=float)
    # Metadata
    index: int = attrs.field(converter=int)
    dofs: tuple[int, int, int] = attrs.field(init=False)
    # Boundary conditions
    fix_u: bool = attrs.field(converter=_parse_bool_from_str, default=False)
    fix_v: bool = attrs.field(converter=_parse_bool_from_str, default=False)
    fix_theta: bool = attrs.field(converter=_parse_bool_from_str, default=False)


    @dofs.default
    def _set_dofs(self) -> tuple[int, int, int]:
        dofs = tuple(self.index * 3 + i for i in range(3))
        return dofs


def nodes_from_csv(file_path: str) -> list[Node]:
    """Serialize ``Node`` instances from a CSV file."""
    nodes: list[Node] = []
    existing_indices: set[int] = set()
    with open(file_path) as file_buffer:
        reader = csv.DictReader(file_buffer)
        for record in reader:
            if record["index"] in existing_indices:
                _LOGGER.warning(f"Duplicate node index: {record["index"]}")
            node = Node(**record)
            existing_indices.add(node.index)
            nodes.append(node)
    return nodes


@attrs.define
class BeamElement:
    """A 2D Bernoulli-Euler beam element."""

    start_node: Node
    end_node: Node
    index: int = attrs.field(converter=int)
    modulus_of_elasticity: float = attrs.field(converter=float)
    moment_of_inertia: float = attrs.field(converter=float)
    area: float = attrs.field(converter=float)
    length: float = attrs.field(init=False)
    angle: float = attrs.field(init=False)

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


def elements_from_csv(file_path: str, nodes: list[Node]):
    """Serialize ``BeamElement`` instances from CSV."""
    elements: list[BeamElement] = []

    node_map = {node.index: node for node in nodes}

    existing_indices: set[int] = set()
    with open(file_path) as file_buffer:
        reader = csv.DictReader(file_buffer)
        for record in reader:
            if record["index"] in existing_indices:
                _LOGGER.warning(f"Duplicate node index: {record["index"]}")
            element = BeamElement(
                index=record["index"],
                start_node=node_map[int(record["start_node"])],
                end_node=node_map[int(record["end_node"])],
                modulus_of_elasticity=record["modulus_of_elasticity"],
                moment_of_inertia=record["moment_of_inertia"],
                area=record["area"],
            )
            existing_indices.add(element.index)
            elements.append(element)
    return elements
