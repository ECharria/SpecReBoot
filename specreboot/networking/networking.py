from typing import Optional
import pandas as pd
import networkx as nx

# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------

def _validate_matrix_pair(df_a: pd.DataFrame, df_b: pd.DataFrame) -> None:
    """Validate that two DataFrames are square and aligned."""
    if not isinstance(df_a, pd.DataFrame) or not isinstance(df_b, pd.DataFrame):
        raise TypeError("Both inputs must be pandas DataFrames.")

    if df_a.shape != df_b.shape:
        raise ValueError("Mean similarity and support matrices must have same shape.")

    if df_a.index.tolist() != df_b.index.tolist():
        raise ValueError("DataFrame indices must match.")

    if df_a.columns.tolist() != df_b.columns.tolist():
        raise ValueError("DataFrame columns must match.")


# ----------------------------------------------------------------------
# Component filtering
# ----------------------------------------------------------------------

def _prune_component_edges(
    graph: nx.Graph,
    component_nodes,
    cosine_delta: float
) -> None:
    """
    Remove the lowest-weight edges within a component until it shrinks.
    """
    # Extract all edges inside this component
    sub_edges = graph.subgraph(component_nodes).edges(data=True)

    # Sort by edge weight (mean_similarity)
    sorted_edges = sorted(sub_edges, key=lambda e: e[2].get("weight", 0.0))

    if not sorted_edges:
        return

    # Remove the weakest edge
    weakest_edge = sorted_edges[0]
    u, v = weakest_edge[0], weakest_edge[1]
    graph.remove_edge(u, v)


def _filter_components(
    graph: nx.Graph,
    max_component_size: int,
    cosine_delta: float
) -> None:
    """
    Iteratively prune oversized components by removing weakest edges.

    Parameters
    ----------
    graph : nx.Graph
        The network to be filtered.
    max_component_size : int
        Maximum allowed connected component size (0 = disabled).
    cosine_delta : float
        Placeholder for future logic; currently not used directly.
    """
    if max_component_size == 0:
        return

    oversized_exists = True

    while oversized_exists:
        oversized_exists = False
        components = list(nx.connected_components(graph))

        for comp in components:
            if len(comp) > max_component_size:
                _prune_component_edges(graph, comp, cosine_delta)
                oversized_exists = True

    # Assign component ID numbers
    for cid, comp in enumerate(nx.connected_components(graph)):
        for node in comp:
            graph.nodes[node]["component"] = cid


def build_base_graph(
    df_mean_sim: pd.DataFrame,
    df_support: pd.DataFrame,
    sim_threshold: float = 0.7,
    max_component_size: int | None = None,
    cosine_delta: float = 0.02,
    output_file: str = "network_similarity.graphml",
) -> nx.Graph:
    """
    Build a graph where edges exist if mean similarity exceeds a threshold.
    Component filtering is optional: enabled only if max_component_size is not None.
    """
    _validate_matrix_pair(df_mean_sim, df_support)

    G = nx.Graph()
    scan_ids = df_mean_sim.index.tolist()
    G.add_nodes_from(scan_ids)

    n = len(scan_ids)
    for i in range(n):
        for j in range(i + 1, n):
            sim = df_mean_sim.iloc[i, j]
            if sim >= sim_threshold:
                sup = df_support.iloc[i, j]
                G.add_edge(
                    scan_ids[i], scan_ids[j],
                    weight=sim,
                    bootstrap_support=sup
                )

    # Only filter if parameter is provided
    if max_component_size is not None:
        _filter_components(G, max_component_size, cosine_delta)

    nx.write_graphml(G, output_file)
    return G




# Function 2: Dual-threshold similarity + support graph
def build_thresh_graph(
    df_mean_sim: pd.DataFrame,
    df_support: pd.DataFrame,
    sim_threshold: float = 0.7,
    support_threshold: float = 0.3,
    max_component_size: int | None = None,
    cosine_delta: float = 0.02,
    output_file: str = "network_supported.graphml",
) -> nx.Graph:
    """
    Build a network where edges require both:
        - similarity ≥ sim_threshold
        - bootstrap support ≥ support_threshold
    Component filtering is optional.
    """
    _validate_matrix_pair(df_mean_sim, df_support)

    G = nx.Graph()
    scan_ids = df_mean_sim.index.tolist()
    G.add_nodes_from(scan_ids)

    n = len(scan_ids)
    for i in range(n):
        for j in range(i + 1, n):
            sim = df_mean_sim.iloc[i, j]
            sup = df_support.iloc[i, j]
            if sim >= sim_threshold and sup >= support_threshold:
                G.add_edge(
                    scan_ids[i], scan_ids[j],
                    weight=sim,
                    bootstrap_support=sup
                )

    if max_component_size is not None:
        _filter_components(G, max_component_size, cosine_delta)

    nx.write_graphml(G, output_file)
    return G




# Function 3: Multi-class (core + rescued edges)
def build_core_rescue_graph(
    df_mean_sim: pd.DataFrame,
    df_support: pd.DataFrame,
    sim_core: float = 0.7,
    support_core: float = 0.3,
    sim_rescue_min: float = 0.2,
    support_rescue: float = 0.4,
    max_component_size: int | None = None,
    cosine_delta: float = 0.02,
    output_file: str = "network_multiclass.graphml",
) -> nx.Graph:
    """
    Build a graph with two edge classes:
        - core
        - rescued
    Component filtering is optional.
    """
    _validate_matrix_pair(df_mean_sim, df_support)

    G = nx.Graph()
    scan_ids = df_mean_sim.index.tolist()
    G.add_nodes_from(scan_ids)

    n = len(scan_ids)
    for i in range(n):
        for j in range(i + 1, n):
            sim = df_mean_sim.iloc[i, j]
            sup = df_support.iloc[i, j]

            if sim >= sim_core and sup >= support_core:
                label = "core"
            elif sim_rescue_min <= sim < sim_core and sup >= support_rescue:
                label = "rescued"
            else:
                continue

            G.add_edge(
                scan_ids[i], scan_ids[j],
                weight=sim,
                bootstrap_support=sup,
                edge_class=label,
            )

    if max_component_size is not None:
        _filter_components(G, max_component_size, cosine_delta)

    nx.write_graphml(G, output_file)
    return G


