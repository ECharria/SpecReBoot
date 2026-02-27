# specreboot/run_workflow_matchms.py
import argparse
import pickle
import time
from pathlib import Path

from matchms.importing import load_from_mgf
from matchms.similarity.FlashSimilarity import FlashSimilarity

from specreboot.preprocessing.filtering import general_cleaning
from specreboot.binning.binning import global_bins as make_global_bins, bin_spectra
from specreboot.bootstrapping.bootstrapping import calculate_boostrapping
from specreboot.networking.networking import build_base_graph, build_thresh_graph, build_core_rescue_graph


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
            default="Res_matchms",
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
    p.add_argument(
        "--similarities",
        nargs="+",
        default=["flash_cosine", "flash_modcosine", "spec2vec", "ms2deepscore"],
        choices=["all", "flash_cosine", "flash_modcosine", "spec2vec", "ms2deepscore"],
        help=(
            "Which similarity metrics to run. "
            "Use: --similarities all (runs all), or list one/two metrics, e.g. "
            "--similarities flash_modcosine, or --similarities flash_cosine spec2vec."
        ),
    )
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
                "How to label nodes in outputs:\n"
                "  - feature: use 'feature_id' (preferred for feature tables)\n"
                "  - scan: use scan number (if present)\n"
                "  - internal: use internal sequential ids\n"
                "This affects node identifiers in CSV/GraphML and label maps."
            ),
        )

    p.add_argument(
        "--sim-threshold",
        type=float,
        default=0.7,
        help=(
            "Similarity threshold (mean similarity across bootstraps) for edge inclusion "
            "when building graphs for cosine/modcosine/spec2vec.\n"
            "Edges below this similarity are excluded regardless of support."
        ),
    )    
    p.add_argument(
        "--sim-threshold-ms2dp",
        type=float,
        default=0.8,
        help=(
            "Similarity threshold (mean similarity) specifically for MS2DeepScore graphs. "
            "MS2DeepScore often uses a higher cutoff than cosine-based scores."
        ),
    )

    p.add_argument(
            "--tolerance",
            type=float,
            default=0.01,
            help=(
                "Fragment m/z tolerance (Da) for FlashSimilarity matching. "
                "Used for both flash_cosine (fragment) and flash_modcosine (hybrid)."
            ),
        )
    
    p.add_argument(
        "--support-threshold",
        type=float,
        default=0.5,
        help=(
            "Minimum bootstrap edge support for the 'threshold' graph.\n"
            "Support is typically the fraction of bootstraps where an edge appears among top-k.\n"
        ),
    )
    p.add_argument(
        "--max-component-size",
        type=int,
        default=100,
        help=(
            "Maximum allowed connected-component size in the 'threshold' graph. "
            "Components larger than this are trimmed according to your networking rules "
            "(prevents giant hairballs)."
        ),
    )
    p.add_argument(
            "--support-core",
            type=float,
            default=0.5,
            help=(
                "Bootstrap support threshold for defining 'core' edges in the rescue graph. "
                "Core edges are high-support anchors used to rescue additional edges."
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
        "--support-rescue",
        type=float,
        default=0.5,
        help=(
            "Bootstrap support threshold for edges considered eligible for rescue into the core. "
            "Often equal to --support-core, but can be tuned independently."
        ),
    )

def _resolve_and_validate_similarities(args) -> list[str]:
    sims = list(args.similarities)

    # Expand "all"
    if "all" in sims:
        sims = ["flash_cosine", "flash_modcosine", "spec2vec", "ms2deepscore"]

    # De-duplicate while preserving order
    seen = set()
    sims = [s for s in sims if not (s in seen or seen.add(s))]

    # Conditional requirements
    if "ms2deepscore" in sims and args.ms2dp_model is None:
        raise SystemExit("ERROR: --ms2dp-model is required when --similarities includes ms2deepscore")

    if "spec2vec" in sims and args.spec2vec_model is None:
        raise SystemExit("ERROR: --spec2vec-model is required when --similarities includes spec2vec")

    return sims


def calculate_similarities(binned_spectra, bins, model_name: str, similarity, args, outdir: Path):
    df_mean_sim, df_edge_sup, history = calculate_boostrapping(
        binned_spectra,
        bins,
        B=args.B,
        k=args.k,
        similarity_metric=similarity,
        n_jobs=args.n_jobs,
        return_history=True,
        track_bins=True,
        return_label_map=True,
        label_mode=args.label_mode,
    )

    df_mean_sim.to_csv(outdir / f"{args.prefix}_bootstrap_mean_similarity_{model_name}.csv", index=False)
    df_edge_sup.to_csv(outdir / f"{args.prefix}_bootstrap_edge_support_{model_name}.csv", index=False)

    with open(outdir / f"bootstrap_history_{args.prefix}_{model_name}.pkl", "wb") as f:
        pickle.dump(history, f)

    return df_mean_sim, df_edge_sup, history


def networking_score(df_mean_sim, df_edge_sup, similarity_score: str, sim_threshold: float, args, outdir: Path):
    build_base_graph(
        df_mean_sim, df_edge_sup,
        sim_threshold=sim_threshold,
        output_file=str(outdir / f"{args.prefix}_bootstrap_base_{similarity_score}.graphml"),
    )

    build_thresh_graph(
        df_mean_sim, df_edge_sup,
        sim_threshold=sim_threshold,
        max_component_size=args.max_component_size,
        support_threshold=args.support_threshold,
        output_file=str(outdir / f"{args.prefix}_bootstrap_threshold_{similarity_score}.graphml"),
    )

    build_core_rescue_graph(
        df_mean_sim, df_edge_sup,
        sim_core=sim_threshold,
        support_core=args.support_core,
        sim_rescue_min=args.sim_rescue_min,
        support_rescue=args.support_rescue,
        output_file=str(outdir / f"{args.prefix}_bootstrap_rescued_{similarity_score}.graphml"),
    )


def run(args):
    args.outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    spectra = load_from_mgf(str(args.mgf))
    cleaned_name = args.cleaned_mgf or str(args.outdir / f"{args.mgf.stem}_cleaned.mgf")
    spectra_cleaned, report = general_cleaning(spectra, file_name=cleaned_name)
    print(report)

    bins = make_global_bins(spectra_cleaned, args.decimals)
    binned_spectra = bin_spectra(spectra_cleaned, args.decimals)

    similarity_objs = {}

    sim_keys = _resolve_and_validate_similarities(args)


    if "flash_cosine" in sim_keys:
        similarity_objs["Flash_Cosine"] = FlashSimilarity(
            score_type="cosine", matching_mode="fragment", tolerance=args.tolerance
        )

    if "flash_modcosine" in sim_keys:
        similarity_objs["Flash_ModCosine"] = FlashSimilarity(
            score_type="cosine", matching_mode="hybrid", tolerance=args.tolerance
        )

    if "spec2vec" in sim_keys:
        from spec2vec import Spec2Vec
        import gensim

        w2v = gensim.models.Word2Vec.load(str(args.spec2vec_model))
        similarity_objs["Spec2Vec"] = Spec2Vec(
            model=w2v,
            intensity_weighting_power=0.5,
            allowed_missing_percentage=5.0,
        )

    if "ms2deepscore" in sim_keys:
        from ms2deepscore.models import load_model
        from ms2deepscore import MS2DeepScore

        ms2dp_model = load_model(str(args.ms2dp_model))
        similarity_objs["MS2DeepScore"] = MS2DeepScore(ms2dp_model)


    #Run selected metrics
    for model_name, similarity in similarity_objs.items():
        df_mean_sim, df_edge_sup, _ = calculate_similarities(
            binned_spectra,
            bins,
            model_name,
            similarity,
            args,
            args.outdir,
        )

        sim_thr = args.sim_threshold_ms2dp if model_name == "MS2DeepScore" else args.sim_threshold

        networking_score(
            df_mean_sim,
            df_edge_sup,
            model_name,
            sim_thr,
            args,
            args.outdir,
        )

        elapsed = time.time() - t0
        (args.outdir / f"runtime_{args.prefix}.txt").write_text(
            f"Total runtime: {elapsed/60:.2f} min ({elapsed:.1f} s)\n"
        )
        print(f"Total runtime: {elapsed/60:.2f} min ({elapsed:.1f} s)")



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    build_parser(p)
    run(p.parse_args())
