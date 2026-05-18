import networkx as nx
import numpy as np
import pandas as pd
from specreboot.networking.networking import _filter_components, _apply_max_links


def _filter_graph_by_component_size(
    G: nx.Graph,
    max_component_size: int,
    cosine_delta: float,
) -> None:
    """NetworkX adapter for _filter_components.

    Translates the graph's edges into the numpy arrays expected by
    _filter_components, runs the greedy union-find size filtering, then
    removes any rejected edge directly from G.

    Parameters
    ----------
    G : nx.Graph
        The graph to filter. Edges are removed in-place.
    max_component_size : int
        Maximum allowed number of nodes in any connected component.
    cosine_delta : float
        Passed through to _filter_components (reserved, currently unused).

    Returns
    -------
    None
        G is modified in-place; edges that would cause a component to exceed
        max_component_size are removed. No new graph is returned.
    """
    edges = list(G.edges(data=True))
    if not edges:
        return
    node_idx = {n: i for i, n in enumerate(G.nodes())}
    u_nodes = np.array([node_idx[u] for u, _, _ in edges])
    v_nodes = np.array([node_idx[v] for _, v, _ in edges])
    sim_array = np.array([d.get("weight", 0.0) for _, _, d in edges])
    edge_mask = np.ones(len(edges), dtype=bool)
    accepted = _filter_components(
        edge_mask, u_nodes, v_nodes, sim_array, max_component_size, cosine_delta, retire_groups=True
    )
    for (u, v, _), keep in zip(edges, accepted):
        if not keep:
            G.remove_edge(u, v)

def _filter_graph_by_max_links(G: nx.Graph, max_links: int, link_method: str = "mutual") -> None:
    """NetworkX adapter for _apply_max_links. Removes edges in-place."""
    edges = list(G.edges(data=True))
    if not edges:
        return
    node_idx = {n: i for i, n in enumerate(G.nodes())}
    u_nodes = np.array([node_idx[u] for u, _, _ in edges])
    v_nodes = np.array([node_idx[v] for _, v, _ in edges])
    sim_array = np.array([d.get("weight", 0.0) for _, _, d in edges])
    edge_mask = np.ones(len(edges), dtype=bool)
    accepted = _apply_max_links(edge_mask, u_nodes, v_nodes, sim_array, max_links, link_method)
    for (u, v, _), keep in zip(edges, accepted):
        if not keep:
            G.remove_edge(u, v)


def _make_node_only_copy(G_gnps: nx.Graph) -> nx.Graph:
    """
    Create a new graph with the same nodes and node attributes as G_gnps,
    but with NO edges.
    """
    G_new = nx.Graph()
    for n, attrs in G_gnps.nodes(data=True):
        G_new.add_node(n, **dict(attrs))
    return G_new

def load_gnps_graph_and_id_map(
    graphml_path: str,
    bootstrap_ids,
    candidate_node_attrs=("feature_id", "scans", "scan", "spectrum_id", "spectrumid"),
):
    # allow passing a single attr as string
    if isinstance(candidate_node_attrs, str):
        candidate_node_attrs = (candidate_node_attrs,)

    G = nx.read_graphml(graphml_path)

    # 1) direct match: bootstrap IDs == graph node IDs
    node_ids = {str(n): n for n in G.nodes()}
    id_map = {str(x): node_ids[str(x)] for x in bootstrap_ids if str(x) in node_ids}
    if id_map:
        return G, id_map

    # 2) match via node attributes (must be non-ambiguous)
    for attr in candidate_node_attrs:
        lookup = {}
        ok = True
        for n, attrs in G.nodes(data=True):
            if attr in attrs and attrs[attr] not in (None, ""):
                key = str(attrs[attr])
                if key in lookup and lookup[key] != n:
                    ok = False
                    break
                lookup[key] = n

        if ok and lookup:
            id_map = {str(x): lookup[str(x)] for x in bootstrap_ids if str(x) in lookup}
            if id_map:
                return G, id_map

    # no mapping found
    return G, {}


def add_threshold_edges_to_gnps_graph(
    G_gnps: nx.Graph,
    df_mean_sim: pd.DataFrame,
    df_support: pd.DataFrame,
    id_map: dict[str, str],
    sim_threshold: float = 0.7,
    support_threshold: float = 0.3,
    max_links: int | None = None,
    link_method: str = "mutual",
    max_component_size: int | None = None,
    cosine_delta: float = 0.02,
    output_file: str = "gnps_plus_threshold.graphml",
) -> nx.Graph:
    """
    Overlay threshold-filtered bootstrap edges onto an existing GNPS GraphML network.

    Logic
    -----
    For any pair (u, v) present in the bootstrap matrices:
      - keep/add edge only if:
            sim >= sim_threshold
        AND support >= support_threshold

    Behavior
    --------
    - If edge already exists in GNPS:
        preserve all GNPS attributes and add/update bootstrap evidence
    - If edge does not exist in GNPS:
        create a new edge labeled as bootstrap-supported

    Attributes written
    ------------------
    For qualifying edges:
      - bootstrap_sim: float
      - bootstrap_support: float
      - bootstrap_class: "threshold"

    For newly added edges:
      - edge_class: "threshold"
      - origin: "bootstrap"
      - weight: sim
    """
    # Basic validation
    if not isinstance(df_mean_sim, pd.DataFrame) or not isinstance(df_support, pd.DataFrame):
        raise TypeError("df_mean_sim and df_support must be pandas DataFrames.")

    if df_mean_sim.shape != df_support.shape:
        raise ValueError("Mean similarity and support matrices must have same shape.")

    if df_mean_sim.index.tolist() != df_support.index.tolist():
        raise ValueError("DataFrame indices must match.")

    if df_mean_sim.columns.tolist() != df_support.columns.tolist():
        raise ValueError("DataFrame columns must match.")

    G = _make_node_only_copy(G_gnps)
    nodes_in_graph = set(G.nodes())

    ids = list(df_mean_sim.index)

    for i in range(len(ids)):
        a = ids[i]
        a_str = str(a)
        if a_str not in id_map:
            continue
        u = id_map[a_str]

        for j in range(i + 1, len(ids)):
            b = ids[j]
            b_str = str(b)
            if b_str not in id_map:
                continue
            v = id_map[b_str]

            sim = float(df_mean_sim.loc[a, b])
            sup = float(df_support.loc[a, b])

            if not (sim >= sim_threshold and sup >= support_threshold):
                continue

            if G.has_edge(u, v):
                G[u][v]["bootstrap_sim"] = sim
                G[u][v]["bootstrap_support"] = sup
                G[u][v]["bootstrap_class"] = "threshold"
            else:
                G.add_edge(
                    u, v,
                    bootstrap_sim=sim,
                    bootstrap_support=sup,
                    bootstrap_class="threshold",
                    edge_class="threshold",
                    origin="bootstrap",
                    weight=sim,
                )

    if max_links is not None:
        _filter_graph_by_max_links(G, max_links, link_method)

    if max_component_size is not None:
        _filter_graph_by_component_size(G, max_component_size, cosine_delta)

    nx.write_graphml(G, output_file)
    return G

def add_rescued_edges_to_gnps_graph(
    G_gnps: nx.Graph,
    df_mean_sim: pd.DataFrame,
    df_support: pd.DataFrame,
    id_map: dict[str, str],
    sim_core: float = 0.7,
    support_core: float = 0.3,
    sim_rescue_min: float = 0.2,
    support_rescue: float = 0.4,
    max_links: int | None = None,
    link_method: str = "mutual",
    max_component_size: int | None = None,
    cosine_delta: float = 0.02,
    output_file: str = "gnps_plus_rescued.graphml",
) -> nx.Graph:
    """
    Overlay bootstrap-derived edges onto an existing GNPS GraphML network.

    What it does
    ------------
    - Keeps the GNPS graph as the base truth (nodes + original edges + all GNPS metadata).
    - Uses your bootstrap tables (mean similarity + edge support) to:
        (A) annotate existing GNPS edges with bootstrap evidence, and/or
        (B) add NEW edges that are "rescued" (or "core") but missing from GNPS.

    Edge classification rules (bootstrap-based)
    ------------------------------------------
    For any pair (u, v) present in your bootstrap tables:
      - If sim >= sim_core AND support >= support_core  -> bootstrap_class = "core"
      - Else if sim_rescue_min <= sim < sim_core AND support >= support_rescue
                                                      -> bootstrap_class = "rescued"
      - Else: ignore (no annotation / no new edge)

    Attributes written
    ------------------
    For edges that qualify:
      - bootstrap_sim: float (from df_mean_sim)
      - bootstrap_support: float (from df_support)
      - bootstrap_class: "core" or "rescued"

    If the edge already exists in GNPS:
      - GNPS attributes are preserved.
      - Only the 3 bootstrap_* attributes above are added/updated.

    If the edge does NOT exist in GNPS and qualifies:
      - A new edge is created with:
          edge_class = bootstrap_class
          origin = "bootstrap"
          weight = bootstrap_sim  (optional but convenient)

    Parameters
    ----------
    G_gnps
        networkx graph read from GNPS graphml.
    df_mean_sim, df_support
        Square DataFrames indexed by bootstrap IDs (same labels on rows/cols).
    id_map
        dict mapping bootstrap IDs (df index values) -> GNPS node IDs in G_gnps.
    sim_core, support_core, sim_rescue_min, support_rescue
        Thresholds for labeling edges as core/rescued.
    output_file
        Path to write the updated GraphML.

    Returns
    -------
    nx.Graph
        Updated graph (GNPS + bootstrap overlay).
    """
    G = _make_node_only_copy(G_gnps)
    nodes_in_graph = set(G.nodes())

    ids = list(df_mean_sim.index)
    for i in range(len(ids)):
        a = str(ids[i])
        if a not in id_map:
            continue
        u = id_map[a]
        if u not in nodes_in_graph:
            continue

        for j in range(i + 1, len(ids)):
            b = str(ids[j])
            if b not in id_map:
                continue
            v = id_map[b]
            if v not in nodes_in_graph:
                continue

            sim = float(df_mean_sim.loc[a, b])
            sup = float(df_support.loc[a, b])

            if sim >= sim_core and sup >= support_core:
                bclass = "core"
            elif sim_rescue_min <= sim < sim_core and sup >= support_rescue:
                bclass = "rescued"
            else:
                continue

            if G.has_edge(u, v):
                G[u][v]["bootstrap_sim"] = sim
                G[u][v]["bootstrap_support"] = sup
                G[u][v]["bootstrap_class"] = bclass
            else:
                G.add_edge(
                    u, v,
                    bootstrap_sim=sim,
                    bootstrap_support=sup,
                    bootstrap_class=bclass,
                    edge_class=bclass,
                    origin="bootstrap",
                    weight=sim,
                )

    if max_links is not None:
        _filter_graph_by_max_links(G, max_links, link_method)

    if max_component_size is not None:
        _filter_graph_by_component_size(G, max_component_size, cosine_delta)

    nx.write_graphml(G, output_file)
    return G
