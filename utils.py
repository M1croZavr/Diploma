from typing import Tuple, Any

import numpy as np
import networkx as nx
from graph_functions import set_layer_attribute_communities, set_layer_attribute_node_filtering,\
    set_layer_attribute_edge_filtering, filter_connected_component


def _preprocess_graph(g: nx.Graph, central_node: Any, radius_threshold: int) -> Tuple[nx.Graph, Any]:
    """
    Preprocessing of the graph. Leaving only the highest connected component of a graph.
    Setting layer attribute: the least number of steps needed to reach central_node from current node.

    Args:
        g: Graph.
        central_node: Central node which is in the focus of layout algorithm.
        radius_threshold : The radius from which we take for calculation only the edges that are part
         of the breadth-first search and ignore the others.

    Returns:
        Preprocessed graph.
    """
    if not isinstance(g, nx.Graph):
        preprocessed_g = nx.Graph()
        preprocessed_g.add_nodes_from(g)
    preprocessed_g = nx.to_undirected(g)
    if not nx.is_connected(preprocessed_g):
        preprocessed_g = filter_connected_component(preprocessed_g)
    else:
        preprocessed_g = preprocessed_g.copy()
    group_to_nodes = set_layer_attribute_node_filtering(preprocessed_g, central_node, radius_threshold)
    return preprocessed_g, group_to_nodes


def rescale_layout(pos, scale=1):
    """Returns scaled position array to (-scale, scale) in all axes.

    The function acts on NumPy arrays which hold position information.
    Each position is one row of the array. The dimension of the space
    equals the number of columns. Each coordinate in one column.

    To rescale, the mean (center) is subtracted from each axis separately.
    Then all values are scaled so that the largest magnitude value
    from all axes equals `scale` (thus, the aspect ratio is preserved).
    The resulting NumPy Array is returned (order of rows unchanged).

    Parameters
    ----------
    pos : numpy array
        positions to be scaled. Each row is a position.

    scale : number (default: 1)
        The size of the resulting extent in all directions.

    Returns
    -------
    pos : numpy array
        scaled positions. Each row is a position.

    See Also
    --------
    rescale_layout_dict
    """
    # Find max length over all dimensions
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(abs(pos[:, i]).max(), lim)
    # rescale to (-scale, scale) in all directions, preserves aspect
    if lim > 0:
        for i in range(pos.shape[1]):
            pos[:, i] *= scale / lim
    return pos


def pol2cart(pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converting polar coordinates to cartesian.

    Args:
        pos: 2-dimensional (n x 2) array with polar coordinates.

    Returns:
        Tuple of 2 arrays: x and y for each data point.
    """
    rho, phi = pos[:, 0], pos[:, 1]
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def cart2pol(pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converting cartesian coordinates to polar.

    Args:
        pos: 2-dimensional (n x 2) array with cartesian coordinates.

    Returns:
        Tuple of 2 arrays: radius and angle for each data point.
    """
    x, y = pos[:, 0], pos[:, 1]
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def calculate_orbital_adjust_coefficients(g: nx.Graph, pos: np.ndarray) -> dict:
    """
    Assign orbital uniformity coefficient to every node in a graph.
    Args:
        g: Graph with 'layer' attribute.
        pos: Current positions of every node in cartesian coordinates.
    Returns:
        Dictionary mapping an orbital to uniformity coefficient.
    """
    orbital_to_coefficient = dict()
    layers = nx.get_node_attributes(g, "layer").values()
    unique_layers = set(layers)
    for layer in unique_layers:
        layer_pos = pos[[i for i, node_layer in enumerate(layers) if node_layer == layer]]
        rho, phi = cart2pol(layer_pos)
        hist, bin_edges = np.histogram(phi)
        orbital_to_coefficient[layer] = np.std(hist)
    return orbital_to_coefficient


def assign_group_coords(graph, pos, group_to_nodes):
    for node in graph.nodes:
        if group := graph.nodes[node].get("group"):
            node_pos = pos[node]
            if node_pos[0] > 0 and node_pos[1] > 0:
                for grouped_node in group_to_nodes[group]:
                    pos[grouped_node] = node_pos + np.random.uniform(0.5, 1, 2)
            elif node_pos[0] > 0 > node_pos[1]:
                for grouped_node in group_to_nodes[group]:
                    adj_pos = np.random.uniform(0.5, 1, 2)
                    adj_pos[1] *= -1
                    pos[grouped_node] = node_pos + adj_pos
            elif node_pos[0] < 0 < node_pos[1]:
                for grouped_node in group_to_nodes[group]:
                    adj_pos = np.random.uniform(0.5, 1, 2)
                    adj_pos[0] *= -1
                    pos[grouped_node] = node_pos + adj_pos
            else:
                for grouped_node in group_to_nodes[group]:
                    adj_pos = -1 * np.random.uniform(0.5, 1, 2)
                    pos[grouped_node] = node_pos + adj_pos
