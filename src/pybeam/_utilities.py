import numpy as np


def pprint_array(array: np.ndarray):
    with np.printoptions(linewidth=999, suppress=True):
        print(np.array_str(array))
