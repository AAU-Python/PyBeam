"""Tests for the node datamodel."""
from pathlib import Path

import pytest

from pybeam.datamodels import Node, nodes_from_csv


@pytest.mark.parametrize("index, expected", [(0, (0, 1, 2)), (1, (3, 4, 5))])
def test_dof(index: int, expected: tuple[int, int, int]):
    """Assert that the DOF numbers are correctly inferred from the node's index."""
    node = Node(1.0, 2.0, index)
    assert node.dofs == expected


def test_from_csv():
    """Test serialization of ``Node`` from a CSV file."""
    file_path = Path(__file__).parent / "data" / "nodes.csv"

    nodes = nodes_from_csv(file_path)

    assert nodes == [
        Node(0.0, 0.0, 0),
        Node(1.0, 0.0, 1),
        Node(0.0, 2.0, 2),
    ]
