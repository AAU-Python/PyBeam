import math

import pytest
import scipy.linalg as la

from pybeam.datamodels import BeamElement, Node
from pybeam.system_matrices import create_mass_matrix, create_stiffness_matrix


@pytest.mark.skip
def test_cantilever_beam():
    """Assert that the correct transverse displacement for a cantilever beam is obtained."""

    element = BeamElement(
        start_node=Node(
            0.0,
            0.0,
            index=0,
            fix_u=True,
            fix_v=True,
            fix_theta=True,
        ),
        end_node=Node(
            5.0,
            0.0,
            index=1,
            fix_u=False,
            fix_v=False,
            fix_theta=False,
        ),
        modulus_of_elasticity=2.1e11,
        moment_of_inertia=8.33e-8,
        area=1e-4,
        index=0,
        density=7850,
    )

    stiffness_matrix = create_stiffness_matrix(element)[3:, 3:]
    mass_matrix = create_mass_matrix(element)[3:, 3:]

    (eigenvalues, eigenvectors) = la.eig(stiffness_matrix, mass_matrix)

    eigenfrequencies = [math.sqrt(eigval) for eigval in eigenvalues]
    # 1 / (2pi * L^2) * sqrt(E I/rho A)
    factor = math.sqrt(element.modulus_of_elasticity * element.moment_of_inertia / (element.density * element.area))
    analytical_result = (
        (1.87510407 / element.length) ** 2 * factor,
        (4.69409113 / element.length) ** 2 * factor,
        (7.85475744 / element.length) ** 2 * factor,
    )

    # assert round(analytical_result, 5) == round(displacement_vector[1], 5)
