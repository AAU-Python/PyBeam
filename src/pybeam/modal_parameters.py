import logging

import numba
import numpy as np
import scipy.linalg as la

_LOGGER = logging.getLogger(__name__)


def get_modal_parameters(stiffness: np.ndarray, mass: np.ndarray, normalize: bool = False):
    """Get undamped eigenfrequencies and modeshapes.

    Returns:
        np.ndarray: Real part of eigenfrequencies.
        np.ndarray: Eigenvectors. Mass-normalized if ``normalize=True``.
    """
    # TODO: add damping ratios
    eigenvalues, eigenvectors = la.eig(stiffness, mass)

    sorting_mask = sorted(range(len(eigenvalues)), key=lambda x: eigenvalues[x])

    eigenfrequencies = np.sqrt(eigenvalues[sorting_mask])
    eigenvectors = eigenvectors[:, sorting_mask]

    modeshapes = _normalize_eigenvectors(eigenvectors, mass)

    # Mass, normalize

    if eigenfrequencies.imag.any():
        _LOGGER.warning("System has complex-valued undamped eigenvalues.")

    return eigenfrequencies.real, modeshapes


# @numba.jit(nopython=True, cache=True)
def _normalize_eigenvectors(eigenvectors: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Mass-normalize eigenvectors and normalize with respect to largest value per vector."""
    normalization_factors = np.sqrt(np.diag(eigenvectors.T.dot(mass).dot(eigenvectors)))

    modeshapes = eigenvectors.dot(np.diag(1 / normalization_factors))

    # for i, vector in enumerate(modeshapes.T):
    #     modeshapes[:, i] = vector / np.where(-np.min(vector) > np.max(vector), np.min(vector), np.max(vector))

    return modeshapes
