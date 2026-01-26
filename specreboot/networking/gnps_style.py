import networkx as nx
import pandas as pd


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



def add_rescued_edges_to_gnps_graph(
    G_gnps: nx.Graph,
    df_mean_sim: pd.DataFrame,
    df_support: pd.DataFrame,
    id_map: dict[str, str],
    sim_core: float = 0.7,
    support_core: float = 0.3,
    sim_rescue_min: float = 0.2,
    support_rescue: float = 0.4,
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
    G = G_gnps.copy()
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

    nx.write_graphml(G, output_file)
    return G
