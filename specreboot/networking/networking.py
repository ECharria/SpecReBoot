import numpy as np
import pandas as pd
import networkx as nx

# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------

def _validate_matrix_pair(df_a: pd.DataFrame, df_b: pd.DataFrame) -> None:
    """Validate that two DataFrames are square and aligned.

    Parameters:
    df_a : pd.DataFrame
        First matrix (e.g. mean similarity).
    df_b : pd.DataFrame
        Second matrix (e.g. bootstrap support).
    """
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

def _filter_components(
    edge_mask: np.ndarray,
    u_nodes: np.ndarray,
    v_nodes: np.ndarray,
    similarity_array: np.ndarray,
    max_component_size: int,
    cosine_delta: float,
    retire_groups: bool,
) -> np.ndarray:
    """Greedily accept edges from strongest to weakest while keeping clusters below a size limit.

    Edges are considered in descending similarity order. An edge is accepted
    only if merging the two clusters it connects would not exceed
    max_component_size. Once a cluster fails to merge (because it would grow
    too large), it is optionally 'retired': subsequent edges that would touch a
    retired cluster are skipped. This prevents a cluster that is already near
    the size limit from later absorbing smaller clusters via weaker edges.

    Cluster membership is tracked with a flat lookup array (union-find style):
    each node stores the ID of its cluster, and the dominant cluster absorbs the
    smaller one on every accepted merge.

    Parameters:
    edge_mask : np.ndarray
        Boolean array of shape (E,) indicating which edges passed the
        similarity/support thresholds and are candidates for inclusion.
    u_nodes : np.ndarray
        Row indices of the upper-triangle edge pairs, shape (E,).
    v_nodes : np.ndarray
        Column indices of the upper-triangle edge pairs, shape (E,).
    similarity_array : np.ndarray
        Similarity value for each edge, shape (E,).
    max_component_size : int
        Maximum number of nodes allowed in a connected component.
    cosine_delta : float
        Reserved parameter; not currently used in filtering logic.
    retire_groups : bool
        If True, clusters that fail a merge are frozen and will not
        participate in any further merges.

    Returns:
    np.ndarray:
        Boolean array of shape (E,) marking which edges are accepted after
        component-size filtering. Combine with the original threshold mask
        using & to get the final edge set.
    """
    retired_groups = set()

    nr_of_nodes = max(np.max(u_nodes), np.max(v_nodes)) + 1
    node_groups = np.arange(nr_of_nodes)   # each node starts in its own singleton cluster
    group_sizes = np.ones(nr_of_nodes)     # every cluster starts with size 1

    # Work on a copy so the caller's array is not modified.
    sim = similarity_array.copy()
    sim[edge_mask == 0] = 0  # zero out edges that did not pass the threshold

    # Process edges from strongest to weakest.
    indices = np.argsort(sim)[::-1]

    mask = np.zeros_like(sim)

    for i in indices:
        strength = sim[i]
        u, v = u_nodes[i], v_nodes[i]

        if strength == 0:  # all remaining edges are zeroed out, nothing left to do
            break

        u_group = node_groups[u]  # look up current cluster of u
        v_group = node_groups[v]  # look up current cluster of v

        if retire_groups and any(g in retired_groups for g in [u_group, v_group]):
            # At least one cluster is frozen; retire both to prevent partial absorption.
            retired_groups.add(u_group)
            retired_groups.add(v_group)
            continue

        if u_group == v_group:
            # Edge is within an existing cluster — no size change, always accept.
            mask[i] = 1
            continue

        u_group_size = group_sizes[u_group]
        v_group_size = group_sizes[v_group]

        if u_group_size + v_group_size > max_component_size:
            # Merging would exceed the limit; retire both clusters.
            retired_groups.add(u_group)
            retired_groups.add(v_group)
            continue

        # Accept the merge and update cluster bookkeeping.
        mask[i] = 1

        # The lower-numbered group absorbs the higher-numbered one.
        dominant_group, purged_group = sorted((u_group, v_group))
        node_groups[node_groups == purged_group] = dominant_group
        group_sizes[dominant_group] = u_group_size + v_group_size
        group_sizes[purged_group] = 0

    return mask.astype(bool)


def _assign_cluster_ids(graph: nx.Graph) -> None:
    """Assign a 'component' attribute to every node, sorted by component size.

    The largest connected component receives id 0, the second largest id 1,
    and so on. The attribute is written directly onto the graph nodes.

    Parameters:
    graph : nx.Graph
        The graph whose nodes will be labelled.
    """
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    for cid, comp in enumerate(components):
        for node in comp:
            graph.nodes[node]["component"] = cid


# ----------------------------------------------------------------------
# Graph builders
# ----------------------------------------------------------------------

def build_base_graph(
    df_mean_sim: pd.DataFrame,
    df_support: pd.DataFrame,
    sim_threshold: float = 0.7,
    max_component_size: int | None = None,
    cosine_delta: float = 0.02,
    output_file: str = "network_similarity.graphml",
) -> nx.Graph:
    """Build a graph where edges exist if mean similarity meets a threshold.

    Each pair of nodes (i, j) receives an edge when their mean bootstrapped
    similarity is at or above sim_threshold. Component-size filtering is
    applied before edges are added to the graph: edges are accepted greedily
    from strongest to weakest, and any edge that would cause a cluster to
    exceed max_component_size is rejected. Filtering is skipped when
    max_component_size is None.

    Parameters:
    df_mean_sim : pd.DataFrame
        Square matrix of mean pairwise bootstrapped similarities.
    df_support : pd.DataFrame
        Square matrix of mutual-kNN bootstrap support values (same shape).
    sim_threshold : float
        Minimum mean similarity required for an edge to be included.
    max_component_size : int or None
        Maximum allowed connected-component size. None disables filtering.
    cosine_delta : float
        Reserved parameter passed to the component filter; not currently used.
    output_file : str
        Path where the resulting graph is written as a GraphML file.

    Returns:
    nx.Graph:
        The constructed graph with 'weight', 'bootstrap_support', and
        'component' attributes on each edge/node.
    """
    _validate_matrix_pair(df_mean_sim, df_support)

    G = nx.Graph()
    scan_ids = [str(x) for x in df_mean_sim.index.tolist()]
    G.add_nodes_from(scan_ids)

    sim = df_mean_sim.values
    sup = df_support.values
    n = len(scan_ids)

    # Extract upper-triangle pairs to avoid counting each edge twice.
    i_idx, j_idx = np.triu_indices(n, k=1)
    sim_vals = sim[i_idx, j_idx]
    sup_vals = sup[i_idx, j_idx]

    mask = sim_vals >= sim_threshold

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


def build_thresh_graph(
    df_mean_sim: pd.DataFrame,
    df_support: pd.DataFrame,
    sim_threshold: float = 0.7,
    support_threshold: float = 0.3,
    max_component_size: int | None = None,
    cosine_delta: float = 0.02,
    output_file: str = "network_supported.graphml",
) -> nx.Graph:
    """Build a graph where edges require both a similarity and a support threshold.

    An edge between nodes (i, j) is included only when their mean bootstrapped
    similarity is at or above sim_threshold AND their mutual-kNN bootstrap
    support is at or above support_threshold. The dual threshold makes this
    graph more conservative than build_base_graph: edges must be both strong
    and reproducible across bootstrap replicates. Component-size filtering
    works identically to build_base_graph.

    Parameters:
    df_mean_sim : pd.DataFrame
        Square matrix of mean pairwise bootstrapped similarities.
    df_support : pd.DataFrame
        Square matrix of mutual-kNN bootstrap support values (same shape).
    sim_threshold : float
        Minimum mean similarity required for an edge to be included.
    support_threshold : float
        Minimum bootstrap support required for an edge to be included.
    max_component_size : int or None
        Maximum allowed connected-component size. None disables filtering.
    cosine_delta : float
        Reserved parameter passed to the component filter; not currently used.
    output_file : str
        Path where the resulting graph is written as a GraphML file.

    Returns:
    nx.Graph:
        The constructed graph with 'weight', 'bootstrap_support', and
        'component' attributes on each edge/node.
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
    """Build a graph that distinguishes core edges from rescued edges.

    Edges fall into one of two classes:
    - core    : similarity >= sim_core AND support >= support_core.
    - rescued : sim_rescue_min <= similarity < sim_core AND support >= support_rescue.

    The rescued class recovers pairs whose similarity is moderate but whose
    bootstrap support is high enough to be considered reliable. Both classes
    compete for the same component-size budget: the greedy filter processes
    all candidate edges (core and rescued combined) in descending similarity
    order, so a rescued edge is only accepted after all stronger core edges
    have already been placed.

    Parameters:
    df_mean_sim : pd.DataFrame
        Square matrix of mean pairwise bootstrapped similarities.
    df_support : pd.DataFrame
        Square matrix of mutual-kNN bootstrap support values (same shape).
    sim_core : float
        Minimum similarity for a core edge.
    support_core : float
        Minimum bootstrap support for a core edge.
    sim_rescue_min : float
        Minimum similarity for a rescued edge (must be < sim_core).
    support_rescue : float
        Minimum bootstrap support for a rescued edge.
    max_component_size : int or None
        Maximum allowed connected-component size. None disables filtering.
    cosine_delta : float
        Reserved parameter passed to the component filter; not currently used.
    output_file : str
        Path where the resulting graph is written as a GraphML file.

    Returns:
    nx.Graph:
        The constructed graph with 'weight', 'bootstrap_support', 'edge_class',
        and 'component' attributes on each edge/node.
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

    core_mask   = (sim_vals >= sim_core)      & (sup_vals >= support_core)
    rescue_mask = (sim_vals >= sim_rescue_min) & (sim_vals < sim_core) & (sup_vals >= support_rescue)
    either_mask = core_mask | rescue_mask

    if max_component_size is not None:
        either_mask &= _filter_components(either_mask, i_idx, j_idx, sim_vals, max_component_size, cosine_delta, retire_groups=True)

    # Determine the class of each surviving edge: core takes priority.
    labels = np.where(core_mask[either_mask], "core", "rescued")

    edges = [
        (scan_ids[i], scan_ids[j], {"weight": float(s), "bootstrap_support": float(p), "edge_class": str(lab)})
        for i, j, s, p, lab in zip(i_idx[either_mask], j_idx[either_mask], sim_vals[either_mask], sup_vals[either_mask], labels)
    ]
    G.add_edges_from(edges)
    _assign_cluster_ids(G)

    nx.write_graphml(G, output_file)
    return G
