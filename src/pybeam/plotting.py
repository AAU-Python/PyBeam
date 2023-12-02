from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from .assembly import reindex_dof
from .datamodels import BeamElement, Node

__all__ = ["plot_structure"]


def set_style():
    """Set the default plotting style."""
    plt.style.use("dark_background")
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.size"] = 12
    plt.rcParams.update({"axes.grid": True, "axes.grid.which": "both"})
    plt.rcParams.update({"grid.alpha": 0.2, "grid.linestyle": "--"})
    plt.rcParams.update(
        {"xtick.top": True, "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True}
    )


def plot_structure(
    elements: list[BeamElement], linecolor="white", linestyle="-", node_labels: bool = True, element_labels: bool = True
):
    """Plot the mesh of beam elements.

    Args:
        elements (list[BeamElement]): List of :py:class:`~pybeam.datamodels.BeamElement` to plot.
        linecolor (str): Color of elements and nodes. Defaults to ``"white"``.
        linestyle (str): Style of element lines. Defaults to ``"-"``.
        node_labels (bool): If ``True``, show node numbering. Defaults to ``True``.
        element_labels (bool): If ``True``, show element numbering. Defaults to ``True``.
    """
    elements = sorted(elements, key=lambda element: element.index)

    plotted_nodes: set[int] = set()

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="datalim")

    for element in elements:
        for node in (element.start_node, element.end_node):
            if node.index not in plotted_nodes:
                _plot_node(node, color=linecolor, labels=node_labels)
                plotted_nodes.add(node.index)
            _plot_element(element, color=linecolor, linestyle=linestyle, labels=element_labels)


def _plot_node(node: Node, color="black", labels: bool = True):
    ax = plt.gca()

    ax.plot(node.x, node.y, marker="o", color=color)
    if labels:
        ax.annotate(
            f"N{node.index}",
            xy=(node.x, node.y),
            xytext=(1.0, 1.0),
            ha="center",
            va="bottom",
            textcoords="offset fontsize",
        )
        node_numbers = tuple(dof if dof is not None else "X" for dof in node.dofs)
        ax.annotate(
            f"{node_numbers}".replace(" ", "").replace("'", ""),
            xy=(node.x, node.y),
            xytext=(1.0, 1.0),
            ha="center",
            va="top",
            size=8,
            textcoords="offset fontsize",
        )


def _plot_element(element: BeamElement, labels: bool = True, color="black", linestyle="-"):
    ax = plt.gca()

    x = (element.start_node.x, element.end_node.x)
    y = (element.start_node.y, element.end_node.y)
    ax.plot(x, y, linestyle=linestyle, color=color)
    if labels:
        ax.annotate(
            f"E{element.index}",
            xy=(sum(x) / 2, sum(y) / 2),
            xytext=(1.0, 1.0),
            ha="center",
            va="bottom",
            textcoords="offset fontsize",
        )


def plot_modeshape(elements: list[BeamElement], modal_matrix: np.ndarray, mode: int = 0, scale=1):
    """Plot modeshape number ``mode``.

    Arguments:
        elements (list[BeamElement]): List of beam elements making up the mesh.
        modal_matrix (np.ndarray): Modal matrix of the structure.
        mode (int): Index of the modeshape, corresponding to the column in the modal matrix, to plot. Defaults to 0.
        scale (int): An arbitrary scaling factor for the modal displacements. Defaults to 1.
    """
    modeshape = modal_matrix[:, mode]

    # NOTE: The deepcopy breaks all references. Therefore, we must process both nodes of an element
    elements_copy = list(map(deepcopy, elements))

    for element in elements_copy:
        for node in (element.start_node, element.end_node):
            if not node.fix_u:
                node.x += modeshape[node.dofs[0]] * scale
            if not node.fix_v:
                node.y += modeshape[node.dofs[1]] * scale

    plot_structure(elements_copy, linecolor="red", node_labels=True, element_labels=True)