import logging
from typing import Literal

import numpy as np

from .datamodels import BeamElement, Node
from .system_matrices import create_rotation_matrix, create_stiffness_matrix

_LOGGER = logging.getLogger(__name__)


def displacements_to_stresses(displacements: np.ndarray, element: BeamElement, z: float):
    """Compute normal and bending stresses for `element` at distance `z` from the neutral axis.

    Args:
        displacements (np.ndarray): Global displacement vector.
        element (BeamElement): The element to compute stresses for.
        z (float): Distance away from neutral axis where bending stress is computed.

    Returns:
        tuple[tuple[float, float], tuple[float, float]]: A tuple of stresses, structured like

        .. code-block:: python

        (
            (normal_stress_start_node, bending_stress_start_node),
            (normal_stress_end_node, bending_stress_end_node)
        )
    """
    stiffness = create_stiffness_matrix(element)
    rotation_matrix = create_rotation_matrix(element)

    element_displacements = np.zeros((6,), dtype=np.float64)
    for i, dof in enumerate((*element.start_node.dofs, *element.end_node.dofs)):
        if dof is not None:
            element_displacements[i] = displacements[dof]

    element_displacements = rotation_matrix.dot(element_displacements)

    reactions = stiffness.dot(element_displacements)

    # We change the sign of the last three reaction to take into account sign convention
    # The element must be in balance, so therefore the axial and transverse reaction at one
    # end are the opposite of those at the other end.
    reactions[3:] *= -1

    # stresses = np.zeros((2,), dtype=np.float64)
    stresses = []
    for i in range(2):
        sigma_b = reactions[i * 3 + 2] / element.moment_of_inertia * z
        sigma_n = reactions[i * 3] / element.area
        # sigma = sigma_b + sigma_n
        stresses.append((sigma_n, sigma_b))
    return tuple(stresses)


def get_stress_history(displacements: np.ndarray, element: BeamElement, z: float, node: Literal["start", "end"]):
    if node == "start":
        index = 0
    elif node == "end":
        index = 1
    else:
        raise ValueError("Argument ``node`` must be 'start' or 'end'.")

    stress_history = np.zeros((displacements.shape[1],), dtype=np.float64)
    for i, x_i in enumerate(displacements.T):
        stresses = displacements_to_stresses(x_i, element, z)

        summed_stresses = sum(stresses[index])

        stress_history[i] = summed_stresses

    return stress_history


def _s_u_1(z: float, L: float):
    """Shape function for first axial DOF."""
    return 1 - z / L


def _s_u_2(z: float, L: float):
    """Shape function for second axial DOF."""
    return z / L


def _s_v_1(z: float, L: float):
    """Shape function for first transverse DOF."""
    return (2 * z**3) / (L**3) - (3 * z**2) / (L**2) + 1


def _s_v_2(z: float, L: float):
    """Shape function for second transverse DOF."""
    return (2 * z**3) / (L**3) + (3 * z**2) / (L**2)


def _s_t_1(z: float, L: float):
    """Shape function for first rotational DOF."""
    return (z**3) / (L**2) - (2 * z**2) / L - z


def _s_t_2(z: float, L: float):
    """Shape function for second rotational DOF."""
    return (z**3) / (L**2) - (z**2) / L


# Strain functions
def _dz_s_u_1(z: float, L: float):
    """First derivative of ``_s_u_1`` with respect to z."""
    return -1 / L


def _dz_s_u_2(z: float, L: float):
    """First derivative of ``_s_u_2`` with respect to z."""
    return 1 / L


def _dzz_s_v_1(z: float, L: float):
    """Second derivative of ``_s_v_1`` with respect to z."""
    return (12 * z) / (L**3) - 6 / (L**2)


def _dzz_s_v_2(z: float, L: float):
    """Second derivative of ``_s_v_2`` with respect to z."""
    return (12 * z) / (L**3) + 6 / (L**2)


def _dzz_s_t_1(z: float, L: float):
    """Second derivative of ``_s_t_1`` with respect to z."""
    return (6 * z) / (L**2) - 4 / L


def _dzz_s_t_2(z: float, L: float):
    """Second derivative of ``_s_v_2`` with respect to z."""
    return (6 * z) / (L**2) - 2 / L
