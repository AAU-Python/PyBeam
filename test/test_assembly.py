import math

import numpy as np
import pytest
import scipy.linalg as la

from pybeam.datamodels import BeamElement
from pybeam.modal_parameters import get_modal_parameters


def test_assembly(elements: list[BeamElement], stiffness: np.ndarray, mass: np.ndarray):
    """Test assembly by checking the eigenfrequencies of a cantilever beam against analytical results."""

    (eigenvalues, _) = la.eig(stiffness, mass)

    eigenfrequencies = sorted([math.sqrt(eigval) for eigval in eigenvalues])
    # 1 / (2pi * L^2) * sqrt(E I/rho A)
    factor = math.sqrt(
        elements[0].modulus_of_elasticity * elements[0].moment_of_inertia / (elements[0].density * elements[0].area)
    )
    analytical_result = (
        (1.87510407 / elements[-1].end_node.x) ** 2 * factor,
        (4.69409113 / elements[-1].end_node.x) ** 2 * factor,
        (7.85475744 / elements[-1].end_node.x) ** 2 * factor,
    )

    for i, frequency in enumerate(analytical_result):
        # Assert that numerical result is within 2% of analytical result
        assert frequency == pytest.approx(eigenfrequencies[i], rel=0.02)


@pytest.mark.skip(reason="WIP")
def test_modal_parameters(stiffness: np.ndarray, mass: np.ndarray):
    frequencies, modeshapes = get_modal_parameters(stiffness, mass)
