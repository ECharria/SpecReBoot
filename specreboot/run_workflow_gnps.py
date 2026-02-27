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
    p.add_argument(
            "--mgf",
            required=True,
            type=Path,
            help=(
                "Input MGF file with MS/MS spectra. "
                "This is the only required input."
            ),
    )    
    p.add_argument(
        "--ms2dp-model",
        type=Path,
        default=None,
        help=(
            "Path to a trained MS2DeepScore model. Required if --similarities includes ms2deepscore."
        ),
    )
    p.add_argument(
            "--spec2vec-model",
            type=Path,
            default=None,
            help=(
                "Path to a trained Spec2Vec Word2Vec model. Required if --similarities includes spec2vec."
            ),
        )

    p.add_argument(
            "--outdir",
            default=Path("."),
            type=Path,
            help=(
                "Output directory where all CSV/PKL/GraphML files will be written. "
                "Created if it does not exist."
            ),
        )
    p.add_argument(
            "--prefix",
            default="Res_GNPS",
            help=(
                "Prefix used to name output files (CSV, PKL, GraphML, runtime log). "
                "Example: --prefix NP2_run1"
            ),
        )
    p.add_argument(
            "--cleaned-mgf",
            default=None,
            help=(
                "Optional path to write the cleaned MGF. "
                "If omitted, a '<input>_cleaned.mgf' file is written into --outdir."
            ),
        )

    # preprocessing/binning/bootstrap
    p.add_argument(
        "--B",
        type=int,
        default=100,
        help=(
            "Number of bootstrap replicates. "
            "Each replicate resamples peaks (within each spectrum) and recomputes similarities. "
            "Higher B = more stable edge-support estimates but slower runtime. "
            "Typical: 30 (quick), 100 (standard), 300+ (high confidence)."
        ),
    )
    p.add_argument(
        "--k",
        type=int,
        default=5,
        help=(
            "Top-k neighbors per node to keep when building candidate edges in each bootstrap replicate "
            "(i.e., for each spectrum keep only its k most similar spectra). "
            "Higher k increases network density and runtime; lower k is stricter/sparser. "
            "Typical: 5–20 depending on dataset size."
        ),
    )
    p.add_argument(
        "--decimals",
        type=int,
        default=2,
        help=(
            "Number of decimals used for m/z binning (global bin grid). "
            "Example: 2 -> 0.01 m/z bins. More decimals = finer bins (potentially sparser); "
            "fewer decimals = coarser bins (more merging)."
        ),
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Number of parallel worker processes/threads used during bootstrapping.",
    )

    p.add_argument(
        "--label-mode",
        default="feature",
        choices=["feature", "scan", "internal"],
        help=(
            "How to label spectra for mapping to GNPS nodes. "
            "'feature' uses the feature ID/name if present; 'scan' uses scan numbers; "
            "'internal' uses the internal order/index. "
        ),
    )
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
    # rescue thresholds (core + rescued edges)
    p.add_argument(
        "--sim-core",
        type=float,
        default=0.7,
        help=(
            "Core-edge similarity threshold. An edge is labeled 'core' if:\n"
            "  mean_similarity >= sim_core  AND  bootstrap_support >= support_core.\n"
            "Typical: 0.6–0.8 (depends on score/metric)."
        ),
    )

    p.add_argument(
        "--support-core",
        type=float,
        default=0.5,
        help=(
            "Core-edge support threshold (bootstrap_support). Used together with --sim-core.\n"
            "Core edges require: bootstrap_support >= support_core.\n"
            "Typical: 0.3–0.8 (higher = stricter, fewer but more reliable edges)."
        ),
    )

    p.add_argument(
        "--sim-rescue-min",
        type=float,
        default=1e-5,
        help=(
            "Minimum similarity for a 'rescued' edge.\n"
            "An edge is labeled 'rescued' if:\n"
            "  sim_rescue_min <= mean_similarity < sim_core  AND  bootstrap_support >= support_rescue.\n"
            "Set close to 0 to allow very low-similarity edges to be rescued if support is high; "
            "increase (e.g., 0.1–0.3) to avoid rescuing extremely weak similarities."
        ),
    )

    p.add_argument(
        "--support-rescue",
        type=float,
        default=0.5,
        help=(
            "Support threshold for 'rescued' edges (bootstrap_support).\n"
            "Rescued edges require: bootstrap_support >= support_rescue AND similarity is below --sim-core "
            "but above --sim-rescue-min.\n"
            "Often set equal to or higher than --support-core to rescue only very stable edges."
        ),
    )
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
