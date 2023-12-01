import logging

import numpy as np
import scipy.linalg as la

_LOGGER = logging.getLogger(__name__)


def get_modal_parameters(stiffness: np.ndarray, mass: np.ndarray, damping: np.ndarray | None = None):
    eigenvalues, eigenvectors = la.eig(stiffness, mass)

    sorting_mask = sorted(range(len(eigenvalues)), key=lambda x: eigenvalues[x])

    eigenfrequencies = np.sqrt(eigenvalues[sorting_mask])
    eigenvectors = eigenvectors[:, sorting_mask]

    modeshapes = _normalize_eigenvectors(eigenvectors, mass)

    # Mass, normalize

    if eigenfrequencies.imag.any():
        _LOGGER.warning("System has complex-valued undamped eigenvalues.")

    return eigenfrequencies.real, modeshapes


def _normalize_eigenvectors(eigenvectors: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Mass-normalize eigenvectors and normalize with respect to largest value per vector."""
    normalization_factors = np.diag(eigenvectors.T.dot(mass).dot(eigenvectors)) / 1

    modeshapes = eigenvectors.dot(np.diag(normalization_factors))

    for i, vector in enumerate(modeshapes.T):
        modeshapes[:, i] = vector / max(vector, key=abs)

    return modeshapes
