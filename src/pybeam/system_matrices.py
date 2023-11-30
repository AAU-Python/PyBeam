import logging

import numpy as np

from .datamodels import BeamElement

_LOGGER = logging.getLogger(__name__)


def create_rotation_matrix(element: BeamElement) -> np.ndarray:
    """Create a rotation matrix for a ``BeamElement``."""
    # See https://media.cheggcdn.com/media/b6b/b6b3e21c-ed57-4a0c-aca6-b078c0245813/phpbpLJEN.png
    cos = np.cos(element.angle)
    sin = np.sin(element.angle)

    matrix = np.array(
        [
            [cos, sin, 0, 0, 0, 0],
            [-sin, cos, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, cos, sin, 0],
            [0, 0, 0, -sin, cos, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )

    return matrix


def create_stiffness_matrix(element: BeamElement):
    """Create a stiffness matrix for a ``BeamElement``."""
    # See https://media.cheggcdn.com/media/b6b/b6b3e21c-ed57-4a0c-aca6-b078c0245813/phpbpLJEN.png
    E = element.modulus_of_elasticity  # noqa E741
    A = element.area  # noqa=E741
    L = element.length  # noqa=E741
    I = element.moment_of_inertia  # noqa=E741

    k_1 = E * A / L
    k_2 = 12 * E * I / L**3
    k_3 = 6 * E * I / L**2
    k_4 = 2 * E * I / L

    matrix = np.array(
        [
            [k_1, 0, 0, -k_1, 0, 0],
            [0, k_2, k_3, 0, -k_2, k_3],
            [0, k_3, 2 * k_4, 0, -k_3, k_4],
            [-k_1, 0, 0, k_1, 0, 0],
            [0, -k_2, -k_3, 0, k_2, -k_3],
            [0, k_3, k_4, 0, -k_3, 2 * k_4],
        ],
        dtype=np.float64,
    )

    return matrix


def create_mass_matrix(element: BeamElement) -> np.ndarray:
    """Create a consistent mass matrix for a ``BeamElement``."""
    # See https://quickfem.com/wp-content/uploads/IFEM.Ch31.pdf
    rho = element.density
    if not rho:
        _LOGGER.warning(f"No mass density defined for element {element.index}")
    A = element.area  # noqa=E741
    L = element.length  # noqa=E741

    matrix = (
        (rho * A * L)
        / 420
        * np.array(
            [
                [140, 0, 0, 70, 0, 0],
                [0, 156, 22 * L, 0, 54, -13 * L],
                [0, 22 * L, 4 * L**2, 0, 13 * L, -3 * L**2],
                [70, 0, 0, 140, 0, 0],
                [0, 54, 13 * L, 0, 156, -22 * L],
                [0, -13 * L, -3 * L**2, 0, -22 * L, 4 * L**2],
            ]
        )
    )

    return matrix
