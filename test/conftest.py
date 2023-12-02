from pathlib import Path

import numpy as np
import pytest

from pybeam.assembly import assemble_system_matrices
from pybeam.datamodels import BeamElement, elements_from_csv, nodes_from_csv


@pytest.fixture
def elements():
    nodes = nodes_from_csv(Path(__file__).parent / "data" / "nodes.csv")

    csv_path = Path(__file__).parent / "data" / "elements.csv"

    yield elements_from_csv(csv_path, nodes)


@pytest.fixture
def system_matrices(elements: list[BeamElement]):
    yield assemble_system_matrices(elements)


@pytest.fixture
def stiffness(system_matrices: tuple[np.ndarray, np.ndarray]):
    return system_matrices[0]


@pytest.fixture
def mass(system_matrices: tuple[np.ndarray, np.ndarray]):
    return system_matrices[1]
