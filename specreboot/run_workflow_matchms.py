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
    p.add_argument("--mgf", required=True, type=Path)
    p.add_argument("--ms2dp-model", required=True, type=Path)
    p.add_argument("--spec2vec-model", required=True, type=Path)

    p.add_argument("--outdir", default=Path("."), type=Path)
    p.add_argument("--prefix", default="MSn")
    p.add_argument("--cleaned-mgf", default=None)

    p.add_argument("--decimals", type=int, default=2)
    p.add_argument("--B", type=int, default=100)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--label-mode", default="feature", choices=["feature", "scan", "internal"])

    p.add_argument("--sim-threshold", type=float, default=0.7)
    p.add_argument("--sim-threshold-ms2dp", type=float, default=0.8)

    p.add_argument("--flash-tolerance", type=float, default=0.01)

    p.add_argument("--support-threshold", type=float, default=0.5)
    p.add_argument("--max-component-size", type=int, default=200)
    p.add_argument("--support-core", type=float, default=0.5)
    p.add_argument("--sim-rescue-min", type=float, default=1e-5)
    p.add_argument("--support-rescue", type=float, default=0.5)


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

    flash_cosine_similarity = FlashSimilarity(score_type="cosine", matching_mode="fragment", tolerance=args.flash_tolerance)
    flash_modcosine_similarity = FlashSimilarity(score_type="cosine", matching_mode="hybrid", tolerance=args.flash_tolerance)

    from ms2deepscore.models import load_model
    from ms2deepscore import MS2DeepScore
    ms2dp_model = load_model(str(args.ms2dp_model))
    ms2deepscore_similarity = MS2DeepScore(ms2dp_model)

    from spec2vec import Spec2Vec
    import gensim
    w2v = gensim.models.Word2Vec.load(str(args.spec2vec_model))
    spec2vec_similarity = Spec2Vec(model=w2v, intensity_weighting_power=0.5, allowed_missing_percentage=5.0)

    df_mean_sim_cos, df_edge_sup_cos, _ = calculate_similarities(binned_spectra, bins, "Flash_Cosine", flash_cosine_similarity, args, args.outdir)
    df_mean_sim_modcos, df_edge_sup_modcos, _ = calculate_similarities(binned_spectra, bins, "Flash_ModCosine", flash_modcosine_similarity, args, args.outdir)
    df_mean_sim_s2v, df_edge_sup_s2v, _ = calculate_similarities(binned_spectra, bins, "Spec2Vec", spec2vec_similarity, args, args.outdir)
    df_mean_sim_ms2dp, df_edge_sup_ms2dp, _ = calculate_similarities(binned_spectra, bins, "MS2DeepScore", ms2deepscore_similarity, args, args.outdir)

    networking_score(df_mean_sim_cos, df_edge_sup_cos, "Flash_Cosine", args.sim_threshold, args, args.outdir)
    networking_score(df_mean_sim_modcos, df_edge_sup_modcos, "Flash_ModCosine", args.sim_threshold, args, args.outdir)
    networking_score(df_mean_sim_s2v, df_edge_sup_s2v, "Spec2Vec", args.sim_threshold, args, args.outdir)
    networking_score(df_mean_sim_ms2dp, df_edge_sup_ms2dp, "MS2DeepScore", args.sim_threshold_ms2dp, args, args.outdir)

    elapsed = time.time() - t0
    (args.outdir / f"runtime_{args.prefix}.txt").write_text(f"Total runtime: {elapsed/60:.2f} min ({elapsed:.1f} s)\n")
    print(f"Total runtime: {elapsed/60:.2f} min ({elapsed:.1f} s)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    build_parser(p)
    run(p.parse_args())
