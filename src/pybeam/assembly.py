"""Module for assembling system matrices."""
import itertools
import logging

import numpy as np

from .datamodels import BeamElement, Node
from .system_matrices import create_mass_matrix, create_rotation_matrix, create_stiffness_matrix

_LOGGER = logging.getLogger(__name__)


def assemble_system_matrices(elements: list[BeamElement]):
    """Assemble the global mass and stiffness matrices.

    Returns:
        tuple[np.ndarray, np.ndarray]: Stiffness matrix, mass matrix.
    """
    n_dof = len(set(itertools.chain(*[(*element.start_node.dofs, *element.end_node.dofs) for element in elements])))

    global_stiffness_matrix = np.zeros((n_dof, n_dof), dtype=np.float64)
    global_mass_matrix = np.zeros((n_dof, n_dof), dtype=np.float64)
    for element in elements:
        dof_element = (*element.start_node.dofs, *element.end_node.dofs)

        rotation_matrix = create_rotation_matrix(element)

        # Stiffness
        element_stiffness_matrix = create_stiffness_matrix(element)
        rotated_element_stiffness_matrix = rotation_matrix.T.dot(element_stiffness_matrix).dot(rotation_matrix)
        global_stiffness_matrix[np.ix_(dof_element, dof_element)] += rotated_element_stiffness_matrix

        # Mass
        element_mass_matrix = create_mass_matrix(element)
        rotated_element_mass_matrix = rotation_matrix.T.dot(element_mass_matrix).dot(rotation_matrix)
        global_mass_matrix[np.ix_(dof_element, dof_element)] += rotated_element_mass_matrix

    fixed_dof = _get_fixed_dofs(elements)
    reduced_stiffness_matrix = apply_boundary_conditions(global_stiffness_matrix, fixed_dof)
    reduced_mass_matrix = apply_boundary_conditions(global_mass_matrix, fixed_dof)

    _LOGGER.info(f"Generated stiffness and mass with {reduced_stiffness_matrix.shape[0]} DOF")
    return reduced_stiffness_matrix, reduced_mass_matrix


def apply_boundary_conditions(matrix: np.ndarray, fixed_dof: list[int]):
    """Delete rows and columns correspondig to fixed DOF."""
    return np.delete(np.delete(matrix, fixed_dof, axis=0), fixed_dof, axis=1)


def _get_fixed_dofs(elements: list[BeamElement]):
    """Get list of fixed DOFs."""

    def _from_node(node: Node):
        return tuple(itertools.compress(node.dofs, (node.fix_u, node.fix_v, node.fix_theta)))

    fixed_dof = set()
    for element in elements:
        for node in (element.start_node, element.end_node):
            maybe_fixed_dofs = _from_node(node)
            if maybe_fixed_dofs:
                fixed_dof.update(maybe_fixed_dofs)

    return list(fixed_dof)


def reindex_dof(elements: list[BeamElement]) -> list[BeamElement]:
    """Shift DOF numbering to take into account boundary conditions.

    For example, for a single beam element with a fixed support in the first node, the original numbering would be
    ``(0, 1, 2, 3, 4, 5)``. After reindexing, DOF ``(3, 4, 5)`` will be renumbered to ``(None, None, None, 0, 1, 2)``.
    """
    shift = 0
    handled_nodes: set(int) = set()
    for element in elements:
        for node in (element.start_node, element.end_node):
            if node.index in handled_nodes:
                continue
            dof_node = list(node.dofs)
            for i, is_fixed in enumerate((node.fix_u, node.fix_v, node.fix_theta)):
                if is_fixed:
                    shift += 1
                    dof_node[i] = None
                else:
                    dof_node[i] -= shift
            dof_node = tuple(dof_node)
            _LOGGER.info(f"Reindexing DOF of node {node.index} from {node.dofs} to {dof_node}")
            node.dofs = dof_node
            handled_nodes.add(node.index)
    return elements
