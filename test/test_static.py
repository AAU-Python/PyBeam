import numpy as np

from pybeam.datamodels import BeamElement, Node
from pybeam.system_matrices import create_stiffness_matrix


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
            4.5,
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
    )

    transverse_load = 123456

    stiffness_matrix = create_stiffness_matrix(element)[3:, 3:]
    load_vector = np.array([0, transverse_load, 0])

    # K^-1 f
    displacement_vector = np.dot(np.linalg.inv(stiffness_matrix), load_vector)

    # PL^3 / 3EI
    analytical_result = (
        transverse_load * element.length**3 / (3 * element.modulus_of_elasticity * element.moment_of_inertia)
    )

    assert round(analytical_result, 5) == round(displacement_vector[1], 5)
