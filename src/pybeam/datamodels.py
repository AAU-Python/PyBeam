"""Dataclasses for representing the finite element model."""
import csv
import logging
from typing import Protocol

import attrs
import numpy as np
import scipy.linalg as la

_LOGGER = logging.getLogger(__name__)

__all__ = ["Node", "BeamElement", "nodes_from_csv", "elements_from_csv", "SnCurveDnv", "SnCurveEurocode"]


def _parse_bool_from_str(string: str) -> bool:
    """Parse a boolean from a string.

    Example:
        >>> _parse_bool_from_str("true")
        True
    """
    if isinstance(string, bool):
        return string
    return string == "true"


@attrs.define
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
                _LOGGER.warning(f"Duplicate node index: {record['index']}")  # pylint: disable=inconsistent-quotes
            node = Node(**record)
            existing_indices.add(node.index)
            nodes.append(node)
    _LOGGER.info(f"Got {len(nodes)} nodes from {file_path}")
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
    density: float = attrs.field(converter=float, default=0)
    length: float = attrs.field(init=False)
    angle: float = attrs.field(init=False)

    @length.default
    def _set_length(self) -> float:
        dx = self.end_node.x - self.start_node.x
        dy = self.end_node.y - self.start_node.y
        return la.norm((dx, dy))

    @angle.default
    def _set_angle(self) -> float:
        dx = self.end_node.x - self.start_node.x
        dy = self.end_node.y - self.start_node.y
        angle = np.arctan2(dy, dx)
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
                _LOGGER.warning(f"Duplicate element index: {record['index']}")  # pylint: disable=inconsistent-quotes
            element = BeamElement(
                index=record["index"],
                start_node=node_map[int(record["start_node"])],
                end_node=node_map[int(record["end_node"])],
                modulus_of_elasticity=record["modulus_of_elasticity"],
                moment_of_inertia=record["moment_of_inertia"],
                area=record["area"],
                density=record.get("density", 0.0),
            )
            existing_indices.add(element.index)
            elements.append(element)
    _LOGGER.info(f"Got {len(elements)} elements from {file_path}")
    return elements


class SnCurve(Protocol):
    """Protocol for a generic SN curve."""

    def get_cycles_at_range(self, stress_range: float) -> float:
        """Get the fatigue life in number of cycles, given a stress range.

        Args:
            stress_range (float): The stress range in consistent units (usually Pa).
        """


@attrs.define
class SnCurveDnv(SnCurve):
    """An SN-curve as defined in DNV-RP-C203.

    Example:
        Assume we have a non-tubular joint with wall thickness of 10 mm in free air. Its SN curve is C2.

        We create this SN curve like so:

        .. code-block:: python

            dnv_c2 = SnCurveDnv(
                m_1=3.0,
                log_a_bar_1=12.301,
                log_a_bar_2=15.835,
                limit_at_1e7=58.48,
                k=0.15,
                t=25,  # t = t_ref if t < t_ref
                t_ref=25,
                m_2=5.0,
            )

    """

    log_a_bar_1: float
    """Intercept of SN curve in region 1 with ``log(N)`` axis."""
    log_a_bar_2: float
    """Intercept of SN curve in region 2 with ``log(N)`` axis."""
    m_1: float
    """Inverse negative slope in region 1."""
    t_ref: float
    """Reference thickness.
    
    Equal to 25 mm for bolts and welded connections other than tubular joints. For tubular joints, the reference
    thickness is 32 mm.
    """
    t: float
    """Thickness through which a crack would most likely grow.

    ``t = t_ref`` is used if ``t < t_ref.``
    """
    k: float
    """Thickness exponent."""
    limit_at_1e7: float
    """Fatigue limit at 1E7 cycles."""
    m_2: float = 5.0
    """Inverse negative slope in region 2."""

    def get_cycles_at_range(self, stress_range: float):
        # Determine if stress range is in region 1 or 2
        if stress_range <= self.limit_at_1e7:
            m = self.m_1
            log_a_bar = self.log_a_bar_1
        else:
            m = self.m_2
            log_a_bar = self.log_a_bar_2

        # log_N = log_a_bar - m * np.log10(stress_range * (self.t / self.t_ref) ** self.k)
        log_N = log_a_bar - m * np.log10(stress_range)
        N = 10**log_N

        return N


@attrs.define
class SnCurveEurocode(SnCurve):
    """SN curve as defined in EN 1993 1-9."""

    # Reference: https://www.phd.eng.br/wp-content/uploads/2015/12/en.1993.1.9.2005-1.pdf

    d_sigma_c = attrs.field(type=float)
    """Detail category."""
    d_sigma_d = attrs.field(type=float)
    """Constant amplitude fatigue limit."""
    d_sigma_l = attrs.field(type=float)
    """Cutoff limit."""

    @d_sigma_d.default
    def _set_d_sigma_d(self) -> float:
        return (2 / 5) ** (1 / 3) * self.d_sigma_c

    @d_sigma_l.default
    def _set_d_sigma_l(self) -> float:
        return (1 / 20) ** (1 / 5) * self.d_sigma_d

    def get_cycles_at_range(self, stress_range: float) -> float:
        if stress_range >= self.d_sigma_d:
            m = 3
            sigma_ref = self.d_sigma_c
            const = 2e6
        elif self.d_sigma_d > stress_range >= self.d_sigma_l:
            m = 5
            sigma_ref = self.d_sigma_d
            const = 5e6
        else:
            return np.inf

        N = (const * sigma_ref**m) / (stress_range**m)
        return N
