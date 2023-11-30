import itertools

import numpy as np

from ._utilities import pprint_array
from .datamodels import BeamElement, Node
from .system_matrices import create_mass_matrix, create_rotation_matrix, create_stiffness_matrix


def assemble_system_matrices(elements: list[BeamElement]):
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
