import itertools

import numpy as np
import pytest

from pybeam.datamodels import BeamElement
from pybeam.plotting import reindex_dof


def test_dof_shift(elements: list[BeamElement]):
    """Test that DOF are renumbered correctly, so that fixed DOF are not part of the numbering."""
    elements_shifted = reindex_dof(elements)

    assert list(
        itertools.chain(*[(*element.start_node.dofs, *element.end_node.dofs) for element in elements_shifted])
    ) == [
        None,
        None,
        None,
        0,
        1,
        2,
        0,
        1,
        2,
        3,
        4,
        5,
        3,
        4,
        5,
        6,
        7,
        8,
    ]


