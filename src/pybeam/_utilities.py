"""Utility stuff that's generally useful."""
import numpy as np


def pprint_array(array: np.ndarray):
    """Print an easily readable numpy array."""
    with np.printoptions(linewidth=999, suppress=True):
        print(np.array_str(array))
