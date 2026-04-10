from typing import Optional
import numpy as np
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


def _filter_components(edge_mask: np.array, u_nodes: np.array, v_nodes: np.array, similarity_array: np.matrix, max_component_size: int, cosine_delta: float, retire_groups: bool) -> np.array:
    """"
    creates a mask that removes edges that would cause clusters to grow too big
    """

    retired_groups = set()
    
    nr_of_nodes = max(np.max(u_nodes), np.max(v_nodes)) + 1
    node_groups = np.array(range(nr_of_nodes))  # lookup
    group_sizes = np.ones(nr_of_nodes) 

    similarity_array = similarity_array.copy()  # make a copy to modify
    similarity_array[edge_mask == 0] = 0  # remove all values that have no edges

    indices = np.argsort(similarity_array)[::-1]  # indices of numbers from high to low

    mask = np.zeros_like(similarity_array)

    for i in indices:
        strength = similarity_array[i]
        u, v = u_nodes[i], v_nodes[i]

        if strength == 0:  # encountering a strength of 0 means we're not going to add any more edges on the remaining data, so we can end the process (remember we sorted them by size)
            break

        u_group = node_groups[u]  # look up the group of u
        v_group = node_groups[v]  # look up the group of v

        if retire_groups and any(g in retired_groups for g in [u_group, v_group]):  # if we turned on this setting, we don't touch the retired groups (matches behaviour of original breakup implementation)
            retired_groups.add(u_group)
            retired_groups.add(v_group)  # we need to make sure BOTH groups are retired after a failed connection
            continue

        if u_group == v_group:  # if they're already in the same cluster, the cluster won't grow in size
            mask[i] = 1
            continue

        u_group_size = group_sizes[u_group]  # get the size of group u
        v_group_size = group_sizes[v_group]  # get the size of group v

        group_sum = u_group_size + v_group_size
        if group_sum > max_component_size:  # adding these clusters would exceed the max size
            retired_groups.add(u_group)
            retired_groups.add(v_group)
            continue

        # if we get here, we're allowed to add the clusters
        mask[i] = 1

        # we need to update our group administration
        dominant_group, purged_group = sorted((u_group, v_group))  # determine which group will take over the members of the other (lowest group nr is dominant)

        node_groups[node_groups == purged_group] = dominant_group
        group_sizes[dominant_group] = group_sum
        group_sizes[purged_group] = 0


    return mask.astype(bool)


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
    scan_ids = [str(x) for x in df_mean_sim.index.tolist()]
    G.add_nodes_from(scan_ids)

    sim = df_mean_sim.values
    sup = df_support.values
    n = len(scan_ids)

    i_idx, j_idx = np.triu_indices(n, k=1)
    sim_vals = sim[i_idx, j_idx]
    sup_vals = sup[i_idx, j_idx]

    mask = sim_vals >= sim_threshold
    
    # Only filter if parameter is provided
    if max_component_size is not None:
        mask &= _filter_components(mask, i_idx, j_idx, sim_vals, max_component_size, cosine_delta, retire_groups=True)

    edges = [
        (scan_ids[i], scan_ids[j], {"weight": float(s), "bootstrap_support": float(p)})
        for i, j, s, p in zip(i_idx[mask], j_idx[mask], sim_vals[mask], sup_vals[mask])
    ]
    G.add_edges_from(edges)
    _assign_cluster_ids(G)

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
    scan_ids = [str(x) for x in df_mean_sim.index.tolist()]
    G.add_nodes_from(scan_ids)

    sim = df_mean_sim.values
    sup = df_support.values
    n = len(scan_ids)

    i_idx, j_idx = np.triu_indices(n, k=1)
    sim_vals = sim[i_idx, j_idx]
    sup_vals = sup[i_idx, j_idx]

    mask = (sim_vals >= sim_threshold) & (sup_vals >= support_threshold)

    if max_component_size is not None:
        mask &= _filter_components(mask, i_idx, j_idx, sim_vals, max_component_size, cosine_delta, retire_groups=True)

    edges = [
        (scan_ids[i], scan_ids[j], {"weight": float(s), "bootstrap_support": float(p)})
        for i, j, s, p in zip(i_idx[mask], j_idx[mask], sim_vals[mask], sup_vals[mask])
    ]
    G.add_edges_from(edges)
    _assign_cluster_ids(G)

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
    scan_ids = [str(x) for x in df_mean_sim.index.tolist()]
    G.add_nodes_from(scan_ids)

    sim = df_mean_sim.values
    sup = df_support.values
    n = len(scan_ids)

    i_idx, j_idx = np.triu_indices(n, k=1)
    sim_vals = sim[i_idx, j_idx]
    sup_vals = sup[i_idx, j_idx]

    core_mask   = (sim_vals >= sim_core)       & (sup_vals >= support_core)
    rescue_mask = (sim_vals >= sim_rescue_min)  & (sim_vals < sim_core) & (sup_vals >= support_rescue)
    either_mask = core_mask | rescue_mask

    if max_component_size is not None:
        either_mask &= _filter_components(either_mask, i_idx, j_idx, sim_vals, max_component_size, cosine_delta, retire_groups=True)

    labels = np.where(core_mask[either_mask], "core", "rescued").astype(str).astype(str)

    edges = [
        (scan_ids[i], scan_ids[j], {"weight": float(s), "bootstrap_support": float(p), "edge_class": str(lab)})
        for i, j, s, p, lab in zip(i_idx[either_mask], j_idx[either_mask], sim_vals[either_mask], sup_vals[either_mask], labels)
    ]
    G.add_edges_from(edges)
    _assign_cluster_ids(G)

    nx.write_graphml(G, output_file)
    return G


def _assign_cluster_ids(graph):
    all_nodes = list(nx.connected_components(graph))
    sorted_nodes = sorted(all_nodes, key=len, reverse=True)  # largest cluster has id 0

    for cid, comp in enumerate(sorted_nodes):
        for node in comp:
            graph.nodes[node]["component"] = cid