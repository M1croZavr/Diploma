from typing import Any

import numpy as np
import scipy as sp
import networkx as nx
from utils import _preprocess_graph, calculate_orbital_adjust_coefficients, pol2cart

SERIES_VF = np.vectorize(lambda x: sum(2 / i for i in range(1, x + 1)))
LAYER_TO_RADIUS = {layer: sum(1 / np.log(1 + i) for i in range(1, layer + 1)) for layer in range(101)}


def central_spring_layout(
        graph: nx.Graph,
        central_node: Any,
        pos: dict,
        fixed: list,
        l: float = None,
        radius_threshold: int = 5,
        alpha: float = 1.0,
        iterations: int = 50,
        threshold: float = 1e-4,
        dim: int = 2,
        seed: int = 12345
):
    """
    Position nodes using radius power modified Fruchterman-Reingold force-directed algorithm.

    The algorithm simulates a force-directed representation of the network
    treating edges as springs holding nodes close, while treating nodes
    as repelling objects, sometimes called an antigravity force.
    Simulation continues until the positions are close to an equilibrium.

    There are some hard-coded values: minimal distance between
    nodes (0.01) and "temperature" of 0.1 to ensure nodes don't fly away.
    During the simulation, `l` helps determine the distance between nodes.

    Fixing some nodes doesn't allow them to move in the simulation.

    Args:
    graph : Connected NetworkX graph or list of nodes
        A position will be assigned to every node in g.

    central_node : Any
        Central node identifier relative to which the radius are calculated and layout is built.

    pos : dict
        Initial positions for nodes as a dictionary with node as keys
        and values as a coordinate list or tuple. Required (0, 0) for the central node.

    fixed : list
        Nodes to keep fixed at initial position.
        Nodes not in ``G.nodes`` are ignored.
        ValueError raised if `fixed` specified and `pos` not.
        Required for central node.

    l : float (default=None)
        Optimal distance between nodes.  If None the distance is set to
        1/sqrt(n) where n is the number of nodes.  Increase this value
        to move nodes farther apart. Leave default for stable result.

    radius_threshold : int (default=5)
        The radius from which we take for calculation only the edges that are part of the breadth-first search
        and ignore the others.

    alpha : float (default=1.0)
        Radius based force coefficient.

    iterations : int  optional (default=50)
        Maximum number of iterations taken

    threshold: float optional (default = 1e-4)
        Threshold for relative error in node position changes.
        The iteration stops if the error is below this threshold.

    dim : int
        Dimension of layout.

    seed : int (default=12345)
        Random seed for initial generation.

    Returns:
    pos : dict
        A dictionary of positions keyed by every node in a graph.
    """
    np.random.seed(seed)
    graph, group_to_nodes = _preprocess_graph(graph, central_node, radius_threshold)

    if not pos:
        raise ValueError("at least central node coordinate pair must be provided")
    for node in fixed:
        if node not in pos or node not in graph:
            raise ValueError("nodes are fixed without positions given")
    node_to_index = {node: i for i, node in enumerate(graph)}
    fixed = np.asarray([node_to_index[node] for node in fixed])
    # figure out index number of central node
    central_node = node_to_index[central_node]

    # Determine size of existing domain to adjust initial positions
    dom_size = max(coord for pos_tup in pos.values() for coord in pos_tup)
    if dom_size == 0:
        dom_size = 1
    polar = np.stack(
        (
            np.array([LAYER_TO_RADIUS[layer] for layer in nx.get_node_attributes(graph, "layer").values()]),
            np.random.uniform(low=-1, high=1, size=(len(graph), )) * np.pi
        ),
        axis=1
    )
    cartesian = np.stack(
        pol2cart(polar),
        axis=1
    )
    pos_arr = cartesian * dom_size

    for node in graph:
        if node in pos:
            pos_arr[node_to_index[node]] = np.asarray(pos[node])

    if len(graph) == 0:
        return {}
    if len(graph) == 1:
        return {nx.utils.arbitrary_element(graph.nodes()): np.array([0, 0])}

    # Sparse matrix
    # sparse solver for large graphs
    # adjacency matrix nodes in the same order as graph was constructed
    adjacency = nx.to_scipy_sparse_array(graph, dtype="f")
    if l is None and fixed is not None:
        # We must adjust l by domain size for layouts not near 1x1
        nnodes, _ = adjacency.shape
        l = dom_size / np.sqrt(nnodes)
    pos, poses, central_forces, common_forces = _sparse_fruchterman_reingold(
        adjacency, graph, central_node, l, alpha, pos_arr, fixed, iterations, threshold, dim, seed
    )
    pos = dict(zip(graph, pos))
    return pos, poses, graph, central_forces, common_forces, group_to_nodes


def _sparse_fruchterman_reingold(
        adjacency, graph, central_node, l, alpha, pos, fixed,
        iterations=50, threshold=1e-4, dim=2, seed=12345,
        save_every_iteration=True
):
    np.random.seed(seed)
    try:
        nnodes, _ = adjacency.shape
    except AttributeError as err:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from err
    # make sure we have a LIst of Lists representation
    try:
        adjacency = adjacency.tolil()
    except AttributeError:
        adjacency = sp.sparse.coo_array(adjacency).tolil()

    pos = pos.astype(adjacency.dtype)

    # optimal distance between nodes
    if l is None:
        l = np.sqrt(1.0 / nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    # max side along one of two coordinates
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / (iterations + 1)

    displacement_common = np.zeros((dim, nnodes))
    displacement_central = np.zeros((dim, nnodes))
    poses = []
    central_forces = []
    common_forces = []
    for iteration in range(iterations):
        if save_every_iteration:
            poses.append(pos.copy())
        displacement_common *= 0
        displacement_central *= 0
        orbital_adjust_coefficients = calculate_orbital_adjust_coefficients(graph, pos)
        # loop over rows
        for u, node in zip(range(adjacency.shape[0]), graph):
            if u in fixed:
                continue
            # difference between this row's node position and all others
            delta = (pos[u] - pos).T
            layer = graph.nodes[node]["layer"]
            orbital_coefficient = orbital_adjust_coefficients[layer]
            # distance between points by l2 norm
            distance = np.sqrt((delta ** 2).sum(axis=0))

            # enforce minimum distance of 0.01
            distance = np.where(distance < 0.01, 0.01, distance)
            # the adjacency matrix row
            adjacency_i = adjacency.getrowview(u).toarray()  # TODO: revisit w/ sparse 1D container
            # displacement "force"
            # 2xN * (1xN - 1xN) -> total size = 2x1
            displacement_common[:, u] += (
                    delta * (orbital_coefficient * l ** 2 / distance ** 2 - adjacency_i * distance / l)
            ).sum(axis=1)
            # displacement_common[:, u] += (
            #         delta * (l ** 2 / distance ** 2 - adjacency_i * distance / l)
            # ).sum(axis=1)
            # На первых итерациях алгоритма l2 расстояние совпадает с радиусом поэтому фиктивная сила = 0
            displacement_central[:, u] += alpha * (
                    delta[:, central_node]
                    * (LAYER_TO_RADIUS[layer] ** 2 / distance[central_node] ** 2
                       - distance[central_node] / LAYER_TO_RADIUS[layer])
            )
        central_forces.append(np.mean(displacement_central))
        common_forces.append(np.mean(displacement_common))
        displacement = displacement_common + displacement_central
        # update positions
        length = np.sqrt((displacement ** 2).sum(axis=0))
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = (displacement * t / length).T
        pos += delta_pos
        # cool temperature
        t -= dt
        if (np.linalg.norm(delta_pos) / nnodes) < threshold:
            break
    return pos, poses, central_forces, common_forces
