from collections import namedtuple
import random

import networkx as nx
import numpy as np
import pytest

from specreboot.networking.networking import _filter_components


EdgeData = namedtuple("EdgeData", ["u", "v", "w"])


@pytest.fixture(scope="module")
def random_edge_data():
    """Fixture to generate reproducible mock edge data."""
    rng = random.Random(42)
    edge_dict = {}

    for _ in range(100):
        u = rng.randint(0, 100)
        v = rng.randint(0, 100)
        w = rng.randint(0, 10) / 10  # this simulates many edge cases
        key = tuple(sorted((u, v)))
        edge_dict[key] = w

    u = np.array([k[0] for k in edge_dict])
    v = np.array([k[1] for k in edge_dict])
    w = np.array([w for w in edge_dict.values()])

    u_min = np.minimum(u, v)
    v_max = np.maximum(u, v)

    return EdgeData(u_min, v_max, w)


def build_graph_from_edge_data(edge_data: EdgeData) -> nx.Graph:
    """Helper function to build a NetworkX graph from EdgeData."""
    graph = nx.Graph()
    unique_nodes = set(edge_data.u) | set(edge_data.v)
    for node in unique_nodes:
        graph.add_node(node)

    for a, b, c in zip(edge_data.u, edge_data.v, edge_data.w):
        graph.add_edge(a, b, weight=c)

    return graph


def derive_edge_data(graph: nx.Graph) -> EdgeData:
    """Helper function to extract EdgeData back from a NetworkX graph."""
    u_nodes, v_nodes, weights = [], [], []

    for u, v, data in graph.edges(data=True):
        u_nodes.append(u)
        v_nodes.append(v)
        weights.append(data["weight"])

    return EdgeData(np.array(u_nodes), np.array(v_nodes), np.array(weights))


def run_legacy_filter(graph: nx.Graph, max_component_size: int) -> EdgeData:
    """Runs the legacy NetworkX component filtering logic."""
    
    def _prune_component_edges(graph: nx.Graph, component_nodes) -> None:
        sub_edges = graph.subgraph(component_nodes).edges(data=True)

        # ⚠️ this advanced filtering logic was added to make the function behaviour completely determinstic
        def sort_key(edge):
            u, v, data = edge
            w = data["weight"]
            u, v = tuple(sorted((u, v)))
            return -w, u, v
        # ⚠️ 

        sorted_edges = sorted(sub_edges, key=sort_key, reverse=True)

        if not sorted_edges:
            return

        weakest_edge = sorted_edges[0]
        graph.remove_edge(weakest_edge[0], weakest_edge[1])
        return 


    if max_component_size == 0:
        return derive_edge_data(graph)

    oversized_exists = True
    while oversized_exists:
        oversized_exists = False
        components = list(nx.connected_components(graph))

        for comp in components:
            if len(comp) > max_component_size:
                _prune_component_edges(graph, comp)
                oversized_exists = True

    for cid, comp in enumerate(nx.connected_components(graph)):
        for node in comp:
            graph.nodes[node]["component"] = cid

    return derive_edge_data(graph)


def run_current_filter(edge_data: EdgeData, max_component_size: int) -> EdgeData:
    """Runs the current specreboot `_filter_components` implementation."""
    cosine_delta = 0.0
    edge_mask = np.array([True] * len(edge_data.u))

    mask = _filter_components(
        edge_mask,
        edge_data.u,
        edge_data.v,
        edge_data.w,
        max_component_size,
        cosine_delta,
        retire_groups=True,
    )
    return EdgeData(*(arr[mask] for arr in edge_data))


def normalize_edges(edge_data: EdgeData) -> set:
    """Utility to turn EdgeData arrays into a sorted, hashable set of edge tuples."""
    result = set()
    for u, v in zip(edge_data.u, edge_data.v):
        u, v = sorted((u, v))
        result.add((u, v))
    return result


@pytest.mark.parametrize("max_component_size", [1, 3, 5])
def test_filter_components_matches_legacy(random_edge_data, max_component_size):
    """
    Test that the optimized `_filter_components` implementation outputs
    the exact same edges as the baseline legacy implementation.
    """
    graph = build_graph_from_edge_data(random_edge_data)

    legacy_result = run_legacy_filter(graph, max_component_size)
    current_result = run_current_filter(random_edge_data, max_component_size)

    legacy_edges = normalize_edges(legacy_result)
    current_edges = normalize_edges(current_result)

    assert current_edges == legacy_edges, (
        f"Mismatch found for max_component_size={max_component_size}:\n"
        f"Missing edges: {legacy_edges - current_edges}\n"
        f"Unexpected edges: {current_edges - legacy_edges}"
    )