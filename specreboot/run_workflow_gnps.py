# specreboot/run_workflow_gnps.py
import argparse
from pathlib import Path

from matchms.importing import load_from_mgf
from matchms.similarity.FlashSimilarity import FlashSimilarity

from specreboot.preprocessing.filtering import spectra_harmonization
from specreboot.binning.binning import global_bins as make_global_bins, bin_spectra
from specreboot.bootstrapping.bootstrapping import calculate_bootstrapping
from specreboot.networking.gnps_style import load_gnps_graph_and_id_map, add_threshold_edges_to_gnps_graph, add_rescued_edges_to_gnps_graph


def build_parser(p: argparse.ArgumentParser):
    """Add command-line arguments for the GNPS workflow."""
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
            "--gnps-graphml",
            required=True,
            type=Path,
            help=(
                "Input GNPS GraphML network to which rescued edges will be added."
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
            "--batch-size",
            type=int,
            default=10,
            help=(
                "Number of bootstrap iterations to run in each batch. "
                "This is a trade-off between memory usage and parallelization efficiency."
            ),
    )
    p.add_argument(
            "--n-jobs",
            type=int,
            default=8,
            help=(
                "Number of parallel worker processes/threads used during bootstrapping."
            ),
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
        default="modcosine",
        choices=["cosine", "modcosine", "modcos"],
        help=(
            "Similarity metric to use (implemented with FlashSimilarity).\n"
            "  - cosine: cosine similarity (FlashSimilarity fragment matching)\n"
            "  - modcosine/modcos: modified cosine (FlashSimilarity hybrid matching)"
        ),
    )
    p.add_argument("--tolerance", type=float, default=0.02, help="Tolerance for (mod)cosine")

    # mapping to GNPS nodes
    p.add_argument(
        "--candidate-node-attrs",
        nargs="+",
        default=["shared name"],
        help='GNPS node attribute(s) to match against df_mean_sim.index (e.g., "shared name")',
    )

    p.add_argument(
        "--sim-threshold",
        type=float,
        default=0.7,
        help=(
            "Similarity threshold for the threshold graph and for defining core edges "
            "in the rescued graph."
        ),
    )

    p.add_argument(
        "--support-threshold",
        type=float,
        default=0.5,
        help=(
            "Minimum edge support for the 'threshold' graph.\n"
            "An edge is kept if similarity >= --sim-threshold and "
            "support >= --support-threshold."
        ),
    )

    p.add_argument(
        "--sim-rescue-min",
        type=float,
        default=1e-5,
        help=(
            "Minimum mean similarity required for a rescued edge (even if it connects into core). "
                "This is a safety floor to prevent adding extremely weak similarities."
        ),
    )

    p.add_argument(
        "--max-component-size",
        type=int,
        default=100,
        help=(
            "Maximum allowed connected-component size in the output graph. "
            "Components larger than this are trimmed by removing weakest edges "
            "(prevents giant hairballs)."
        ),
    )


def _make_similarity(args):
    """Create the similarity object selected through the CLI arguments."""
    sim = args.similarity
    sim = args.similarity
    if sim == "modcos":
        sim = "modcosine"

    if sim == "cosine":
        return FlashSimilarity(score_type="cosine", matching_mode="fragment", tolerance=args.tolerance)

    if sim == "modcosine":
        return FlashSimilarity(score_type="cosine", matching_mode="hybrid", tolerance=args.tolerance)

    raise ValueError(f"Unknown similarity: {args.similarity}")

def run(args):
    """Run the full GNPS workflow from spectra loading to graph export."""
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # --- Load and clean spectra ---
    spectra = list(load_from_mgf(str(args.mgf)))
    cleaned_name = args.cleaned_mgf or str(args.outdir / f"{args.mgf.stem}_cleaned.mgf")
    spectra_cleaned, report = spectra_harmonization(spectra, file_name=cleaned_name)
    print(report)

    # --- Bin spectra for bootstrapping ---
    bins = make_global_bins(spectra_cleaned, args.decimals)
    binned_spectra = bin_spectra(spectra_cleaned, args.decimals)

    # --- Create the requested similarity object ---
    similarity = _make_similarity(args)

    # --- Run bootstrapping ----
    result = calculate_bootstrapping(
        binned_spectra,
        bins,
        B=args.B,
        k=args.k,
        similarity_metric=similarity,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        return_history=False,
        track_bins=False,
        return_label_map=True,
        label_mode=args.label_mode,
    )

    df_mean_sim, df_edge_sup, label_map = result

    # --- Export similarity, support, and label-map outputs ---
    df_mean_sim.to_csv(args.outdir / f"{args.prefix}_bootstrap_mean_similarity.csv", index=False)
    df_edge_sup.to_csv(args.outdir / f"{args.prefix}_bootstrap_edge_support.csv", index=False)
    label_map.to_csv(args.outdir / f"{args.prefix}_label_map.csv", index=False)

    # --- Map bootstrap labels back to GNPS node identifiers and add edges to the GNPS graph ---
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

    out_graph_rescued = str(args.outdir / f"{args.prefix}_gnps_plus_rescued.graphml")
    out_graph_thresh = str(args.outdir / f"{args.prefix}_gnps_threshold.graphml")

    # --- Build the threshold GNPS graph ---
    add_threshold_edges_to_gnps_graph(
        G_gnps=gnps_network,
        df_mean_sim=df_mean_sim,
        df_support=df_edge_sup,
        id_map=id_map,
        sim_threshold=args.sim_threshold,
        support_threshold=args.support_threshold,
        max_component_size=args.max_component_size,
        output_file=out_graph_thresh,
    )

    # --- Build the rescued GNPS graph ---
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
        output_file=out_graph_rescued,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    build_parser(p)
    run(p.parse_args())