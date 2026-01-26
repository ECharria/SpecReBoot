# specreboot/run_workflow_gnps.py
import argparse
import time
from pathlib import Path

from matchms.importing import load_from_mgf
from matchms.similarity import ModifiedCosine, CosineGreedy
from matchms.similarity.FlashSimilarity import FlashSimilarity

from specreboot.preprocessing.filtering import general_cleaning
from specreboot.binning.binning import global_bins as make_global_bins, bin_spectra
from specreboot.bootstrapping.bootstrapping import calculate_boostrapping
from specreboot.networking.gnps_style import load_gnps_graph_and_id_map, add_rescued_edges_to_gnps_graph


def build_parser(p: argparse.ArgumentParser):
    p.add_argument("--mgf", required=True, type=Path, help="MGF used to build df_mean_sim/df_edge_sup")
    p.add_argument("--gnps-graphml", required=True, type=Path, help="GNPS network graphml (original)")
    p.add_argument("--outdir", default=Path("."), type=Path)
    p.add_argument("--prefix", default="GNPS")

    # preprocessing/binning/bootstrap
    p.add_argument("--cleaned-mgf", default=None)
    p.add_argument("--decimals", type=int, default=2)
    p.add_argument("--B", type=int, default=100)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--label-mode", default="feature", choices=["feature", "scan", "internal"])

    # similarity choice (keep simple but flexible)
    p.add_argument(
        "--similarity",
        default="modcos",
        choices=["modcos", "cosine", "flash_cosine", "flash_modcosine"],
        help="Similarity for bootstrapping",
    )
    p.add_argument("--tolerance", type=float, default=0.02, help="Tolerance for (mod)cosine")
    p.add_argument("--flash-tolerance", type=float, default=0.01, help="Tolerance for FlashSimilarity")

    # mapping to GNPS nodes
    p.add_argument(
        "--candidate-node-attrs",
        nargs="+",
        default=["shared name"],
        help='GNPS node attribute(s) to match against df_mean_sim.index (e.g., "shared name")',
    )

    # rescue thresholds (pass through)
    p.add_argument("--sim-core", type=float, default=0.7)
    p.add_argument("--support-core", type=float, default=0.5)
    p.add_argument("--sim-rescue-min", type=float, default=1e-5)
    p.add_argument("--support-rescue", type=float, default=0.5)

    p.add_argument("--output-graphml", default=None, help="Output GNPS+rescued graphml filename")


def _make_similarity(args):
    if args.similarity == "modcos":
        return ModifiedCosine(tolerance=args.tolerance)
    if args.similarity == "cosine":
        return CosineGreedy(tolerance=args.tolerance)
    if args.similarity == "flash_cosine":
        return FlashSimilarity(score_type="cosine", matching_mode="fragment", tolerance=args.flash_tolerance)
    if args.similarity == "flash_modcosine":
        return FlashSimilarity(score_type="cosine", matching_mode="hybrid", tolerance=args.flash_tolerance)
    raise ValueError(f"Unknown similarity: {args.similarity}")


def run(args):
    args.outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    spectra = list(load_from_mgf(str(args.mgf)))
    cleaned_name = args.cleaned_mgf or str(args.outdir / f"{args.mgf.stem}_cleaned.mgf")
    spectra_cleaned, report = general_cleaning(spectra, file_name=cleaned_name)
    print(report)

    bins = make_global_bins(spectra_cleaned, args.decimals)
    binned_spectra = bin_spectra(spectra_cleaned, args.decimals)

    similarity = _make_similarity(args)

    df_mean_sim, df_edge_sup = calculate_boostrapping(
        binned_spectra,
        bins,
        B=args.B,
        k=args.k,
        similarity_metric=similarity,
        n_jobs=args.n_jobs,
        return_history=False,
        track_bins=False,
        return_label_map=True,
        label_mode=args.label_mode,
    )

    df_mean_sim.to_csv(args.outdir / f"{args.prefix}_bootstrap_mean_similarity.csv", index=False)
    df_edge_sup.to_csv(args.outdir / f"{args.prefix}_bootstrap_edge_support.csv", index=False)

    gnps_network, id_map = load_gnps_graph_and_id_map(
        str(args.gnps_graphml),
        df_mean_sim.index,
        candidate_node_attrs=args.candidate_node_attrs,
    )

    if not id_map:
        raise ValueError(
            "Could not map bootstrap IDs to GNPS nodes. "
            "Try a different --candidate-node-attrs value (or multiple)."
        )

    out_graph = args.output_graphml or str(args.outdir / f"{args.prefix}_gnps_plus_rescued.graphml")

    # If your helper supports thresholds, pass them. If not, remove these keyword args.
    add_rescued_edges_to_gnps_graph(
        gnps_network,
        df_mean_sim,
        df_edge_sup,
        id_map,
        sim_core=args.sim_core,
        support_core=args.support_core,
        sim_rescue_min=args.sim_rescue_min,
        support_rescue=args.support_rescue,
        output_file=out_graph,
    )

    elapsed = time.time() - t0
    (args.outdir / f"runtime_{args.prefix}.txt").write_text(f"Total runtime: {elapsed/60:.2f} min ({elapsed:.1f} s)\n")
    print(f"Total runtime: {elapsed/60:.2f} min ({elapsed:.1f} s)")
    print(f"Wrote: {out_graph}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    build_parser(p)
    run(p.parse_args())
