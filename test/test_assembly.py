import math
from pathlib import Path

import pytest
import scipy.linalg as la

from pybeam.assembly import assemble_system_matrices
from pybeam.datamodels import elements_from_csv, nodes_from_csv


def test_assembly():
    """Test assembly by checking the eigenfrequencies of a cantilever beam against analytical results."""
    nodes = nodes_from_csv(Path(__file__).parent / "data" / "nodes.csv")

    csv_path = Path(__file__).parent / "data" / "elements.csv"

    elements = elements_from_csv(csv_path, nodes)

    stiffness_matrix, mass_matrix = assemble_system_matrices(elements)

    (eigenvalues, _) = la.eig(stiffness_matrix, mass_matrix)

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
