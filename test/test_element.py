"""Tests for the beam element datamodel."""
import math
from pathlib import Path

import pytest

from pybeam.datamodels import BeamElement, Node, elements_from_csv, nodes_from_csv


@pytest.mark.parametrize("x_0, y_0, x_1, y_1, expected", [(0.0, 0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0, 1.41)])
def test_length(x_0: float, y_0: float, x_1: float, y_1: float, expected: float):
    """Assert that the length of a beam element is correctly calculated based on the node coordinates."""
    element = BeamElement(
        start_node=Node(x_0, y_0, 0),
        end_node=Node(x_1, y_1, 1),
        index=0,
        modulus_of_elasticity=2.1e11,
        moment_of_inertia=8.33e-8,
        area=1e-4,
    )

    assert round(element.length, 2) == expected


@pytest.mark.parametrize("x_0, y_0, x_1, y_1, expected", [(0.0, 0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.0, math.pi / 4)])
def test_angle(x_0: float, y_0: float, x_1: float, y_1: float, expected: float):
    """Assert that the angle of a beam element's neutral axis relative to the global x-axis is correctly calculated."""
    element = BeamElement(
        start_node=Node(x_0, y_0, 0),
        end_node=Node(x_1, y_1, 1),
        index=0,
        modulus_of_elasticity=2.1e11,
        moment_of_inertia=8.33e-8,
        area=1e-4,
    )

    assert round(element.angle, 6) == round(expected, 6)


def test_from_csv():
    """Assert that elements are properly serialized from CSV."""
    nodes = nodes_from_csv(Path(__file__).parent / "data" / "nodes.csv")

    csv_path = Path(__file__).parent / "data" / "elements.csv"

    elements = elements_from_csv(csv_path, nodes)

    assert elements == [
        BeamElement(
            start_node=nodes[0],
            end_node=nodes[1],
            modulus_of_elasticity=2.1e11,
            moment_of_inertia=8.33e-8,
            area=1e-4,
            index=0,
        ),
        BeamElement(
            start_node=nodes[1],
            end_node=nodes[2],
            modulus_of_elasticity=2.1e11,
            moment_of_inertia=8.33e-8,
            area=1e-4,
            index=1,
        ),
        BeamElement(
            start_node=nodes[2],
            end_node=nodes[0],
            modulus_of_elasticity=2.1e11,
            moment_of_inertia=8.33e-8,
            area=1e-4,
            index=2,
        ),
    ]
