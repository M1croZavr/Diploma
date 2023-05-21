import networkx as nx
from collections import deque, defaultdict

from typing import Dict


def set_layer_attribute_communities(g: nx.Graph, source: int, layer_threshold: int = 6) -> None:
    """For given graph set layer(radius) from source node attribute"""
    if source not in g:
        raise ValueError("Source node not in the graph")
    if layer_threshold < 1:
        raise ValueError("Layer threshold must be higher than 0 (default 6)")
    lp_communities = nx.community.label_propagation_communities(g)
    idx_to_community = dict(zip(range(len(lp_communities)), lp_communities))
    layer = 0
    queue = deque([(source, layer), ])
    visited = {source}
    while queue:
        node, layer = queue.popleft()
        if node not in g.nodes:
            continue
        elif layer > layer_threshold:
            for community_idx, community in idx_to_community.items():
                if node in community:
                    node = f'community{community_idx}'
                    visited.add(node)
                    community = {node for node in community if
                                 g.nodes[node].get("layer") is None
                                 or g.nodes[node].get("layer") > layer_threshold}
                    merge_nodes(g, community, node)
                    break
        g.nodes[node]["layer"] = layer
        for neighbor in g.neighbors(node):
            if neighbor not in visited:
                queue.append((neighbor, layer + 1))
                visited.add(neighbor)
    return


def set_layer_attribute_node_filtering(g: nx.Graph, source: int, layer_threshold: int = 6) -> Dict[int, list]:
    """For given graph set layer(radius) from source node attribute"""
    if source not in g:
        raise ValueError("Source node not in the graph")
    if layer_threshold < 1:
        raise ValueError("Layer threshold must be higher than 0 (default 6)")
    # tracking previous for preventing deleting the same edge we came here in undirected graph
    previous = -1
    layer = 0
    group = 0
    queue = deque([(previous, source, layer), ])
    visited = {source}
    nodes_to_remove = []
    while queue:
        previous, node, layer = queue.popleft()
        if layer == layer_threshold:
            g.nodes[node]["group"] = group
            group += 1
        elif layer > layer_threshold:
            g.nodes[node]["group"] = g.nodes[previous]["group"]
            nodes_to_remove.append(node)
        g.nodes[node]["layer"] = layer
        edges_to_remove = []
        for neighbor in g.neighbors(node):
            if neighbor not in visited:
                queue.append((node, neighbor, layer + 1))
                visited.add(neighbor)
            else:
                if layer >= layer_threshold and neighbor != previous:
                    edges_to_remove.append((node, neighbor))
        g.remove_edges_from(edges_to_remove)

    group_to_nodes = defaultdict(list)
    for node in nodes_to_remove:
        group = g.nodes[node]["group"]
        group_to_nodes[group].append(node)
    g.remove_nodes_from(nodes_to_remove)

    return group_to_nodes


def set_layer_attribute_edge_filtering(g: nx.Graph, source: int, layer_threshold: int = 6) -> None:
    """For given graph set layer(radius) from source node attribute"""
    if source not in g:
        raise ValueError("Source node not in the graph")
    if layer_threshold < 1:
        raise ValueError("Layer threshold must be higher than 0 (default 6)")
    # tracking previous for preventing deleting the same edge we came here in undirected graph
    previous = -1
    layer = 0
    queue = deque([(previous, source, layer), ])
    visited = {source}
    nodes_to_remove = []
    while queue:
        previous, node, layer = queue.popleft()
        g.nodes[node]["layer"] = layer
        edges_to_remove = []
        for neighbor in g.neighbors(node):
            if neighbor not in visited:
                queue.append((node, neighbor, layer + 1))
                visited.add(neighbor)
            else:
                if layer >= layer_threshold and neighbor != previous:
                    edges_to_remove.append((node, neighbor))
        g.remove_edges_from(edges_to_remove)

    return


def filter_connected_component(g: nx.Graph):
    """
    Create new instance of a graph, leaving only the biggest connected component

    Args:
        g: Source graph.

    Returns:
        Modified copy of the source graph.
    """
    largest_cc = max(nx.connected_components(g), key=len)
    preprocessed_g = g.subgraph(largest_cc).copy()
    return preprocessed_g


def merge_nodes(g, nodes, new_node):
    """
    Merges the selected `nodes` of the graph G into one `new_node`,
    meaning that all the edges that pointed to or from one of these
    `nodes` will point to or from the `new_node`.
    attr_dict and **attr are defined as in `G.add_node`.
    """

    g.add_node(new_node)  # Add the 'merged' node
    edges_to_add = []

    for n1, n2 in g.edges():
        # For all edges related to one of the nodes to merge,
        # make an edge going to or coming from the `new gene`.
        if n1 in nodes:
            edges_to_add.append((new_node, n2))
        elif n2 in nodes:
            edges_to_add.append((n1, new_node))

    g.add_edges_from(edges_to_add)
    for n in nodes:  # remove the merged nodes
        g.remove_node(n)


def dfs(g: nx.Graph, source: int):
    visited = set()

    def recursive(node: int):
        if node not in visited:
            visited.add(node)
            for neighbor in g.neighbors(node):
                recursive(neighbor)
            visited.remove(node)

    recursive(source)


if __name__ == "__main__":
    graph = nx.karate_club_graph()
    print(len(graph.edges), len(graph.nodes))
