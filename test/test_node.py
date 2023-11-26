"""Tests for the node datamodel."""
import pytest

from pybeam.datamodels import Node


@pytest.mark.parametrize("index, expected", [(0, (0, 1, 2)), (1, (3, 4, 5))])
def test_dof(index: int, expected: tuple[int, int, int]):
    """Assert that the DOF numbers are correctly inferred from the node's index."""
    node = Node(1.0, 2.0, index)
    assert node.dofs == expected
