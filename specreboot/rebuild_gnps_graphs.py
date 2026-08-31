# rebuild_gnps_graphs.py
import argparse
import pandas as pd
from pathlib import Path

from specreboot.networking.gnps_style import (
    load_gnps_graph_and_id_map,
    add_threshold_edges_to_gnps_graph,
    add_rescued_edges_to_gnps_graph,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mean-sim-csv", type=Path, required=True)
    p.add_argument("--edge-sup-csv", type=Path, required=True)
    p.add_argument("--gnps-graphml", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=Path("."))
    p.add_argument("--prefix", default="Res_GNPS")
    p.add_argument("--candidate-node-attrs", nargs="+", default=["id"])
    p.add_argument("--sim-threshold", type=float, default=0.7)
    p.add_argument("--support-threshold", type=float, default=0.5)
    p.add_argument("--sim-rescue-min", type=float, default=1e-5)
    p.add_argument("--max-component-size", type=int, default=100)
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df_mean_sim = pd.read_csv(args.mean_sim_csv, header=0, index_col=None)
    df_mean_sim.index = range(1, len(df_mean_sim) + 1)

    df_edge_sup = pd.read_csv(args.edge_sup_csv, header=0, index_col=None)
    df_edge_sup.index = range(1, len(df_edge_sup) + 1)

    # Sequential scan numbers (1,2,3...) match GraphML node IDs directly
    df_mean_sim.index   = df_mean_sim.index.astype(str)
    df_mean_sim.columns = df_mean_sim.columns.map(lambda x: str(int(x)))
    df_edge_sup.index   = df_edge_sup.index.astype(str)
    df_edge_sup.columns = df_edge_sup.columns.map(lambda x: str(int(x)))

    print("Index sample:", df_mean_sim.index[:5].tolist())

    gnps_network, id_map = load_gnps_graph_and_id_map(
        str(args.gnps_graphml),
        df_mean_sim.index,
        candidate_node_attrs=args.candidate_node_attrs,
    )

    if not id_map:
        raise ValueError(
            "Could not map bootstrap IDs to GNPS nodes. "
            "Try a different --candidate-node-attrs value."
        )

    print(f"Mapped {len(id_map)} bootstrap IDs to GNPS nodes.")

    add_threshold_edges_to_gnps_graph(
        G_gnps=gnps_network,
        df_mean_sim=df_mean_sim,
        df_support=df_edge_sup,
        id_map=id_map,
        sim_threshold=args.sim_threshold,
        support_threshold=args.support_threshold,
        max_component_size=args.max_component_size,
        output_file=str(args.outdir / f"{args.prefix}_gnps_threshold.graphml"),
    )

    add_rescued_edges_to_gnps_graph(
        gnps_network,
        df_mean_sim,
        df_edge_sup,
        id_map,
        sim_core=args.sim_threshold,
        support_core=args.support_threshold,
        sim_rescue_min=args.sim_rescue_min,
        support_rescue=args.support_threshold,
        max_component_size=args.max_component_size,
        output_file=str(args.outdir / f"{args.prefix}_gnps_plus_rescued.graphml"),
    )

    print("Done.")


if __name__ == "__main__":
    main()