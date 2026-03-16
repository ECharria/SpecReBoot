from pathlib import Path
from matchms.importing import load_from_mgf
from matchms.similarity.FlashSimilarity import FlashSimilarity

from specreboot.preprocessing.filtering import general_cleaning
from specreboot.library.library_matching import (
    confidence_aware_match,
    collect_results,
    get_spectrum_id,
    summarize_top_annotation,
    score_candidates,
    restrict_library_to_top_n,
    _run_single_library_bootstrap,
)
from specreboot.binning.binning import global_bins, bin_spectra
import numpy as np

outdir = Path("/lustre/BIF/nobackup/charr003/projects/PostDoc/SpecReBoot_results/lib_match")
outdir.mkdir(parents=True, exist_ok=True)

query_spectra, _ = general_cleaning(
    load_from_mgf("/lustre/BIF/nobackup/charr003/projects/PostDoc/SpecReBoot_results/data/pesticides_test.mgf"),
    file_name=str(outdir / "pesticides_test_cleaned.mgf"),
)
library_spectra, _ = general_cleaning(
    load_from_mgf("/lustre/BIF/nobackup/charr003/projects/PostDoc/SpecReBoot_results/data/GNPS-NP-feature-id.mgf"),
    file_name=str(outdir / "library_cleaned.mgf"),
)

similarity_metric = FlashSimilarity(score_type="cosine", matching_mode="fragment", tolerance=0.01)

# --- Diagnostic: inspect bootstrap score distribution for the first query ---
query = query_spectra[0]
ranked = score_candidates(query, library_spectra, similarity_metric)
top_matches, restricted_library = restrict_library_to_top_n(ranked, library_spectra, 100)
candidate_ids = [m.candidate_id for m in top_matches]

bins = global_bins([query], decimals=2)  # query only
global_bins_arr = np.asarray(bins, dtype="float32")

query_binned = bin_spectra([query], 2)[0]
library_binned = bin_spectra(list(restricted_library), 2)

print(f"\n--- Diagnostic for {get_spectrum_id(query)} ---")
print(f"Original top hit: {top_matches[0].candidate_id}, score: {top_matches[0].original_score:.4f}")
print(f"Query peaks after binning: {query_binned.peaks.mz.size}")
print(f"Global bins (query-only): {len(bins)}")

for b in range(5):
    res = _run_single_library_bootstrap(
        b=b, query_binned=query_binned, library_binned=library_binned,
        candidate_ids=candidate_ids, global_bins_arr=global_bins_arr,
        similarity_metric=similarity_metric, seed=42, score_threshold=0.7,
    )
    top_cid, top_score = res["boot_scores"][0]
    print(f"  replicate {b}: top hit {top_cid}, score {top_score:.4f}")

# --- Main loop ---
results = []
skipped = []

for query in query_spectra:
    qid = get_spectrum_id(query)
    try:
        result = confidence_aware_match(
            query_spectrum=query,
            library_spectra=library_spectra,
            similarity_metric=similarity_metric,
            B=100,
            top_n=100,
            score_threshold=0.7,
            decimals=2,
            seed=42,
        )
        results.append(result)
        print(f"{result.query_id}: {summarize_top_annotation(result)}")
    except ValueError as e:
        print(f"Skipping {qid}: {e}")
        skipped.append(qid)

print(f"\nProcessed {len(results)} spectra, skipped {len(skipped)}.")
if skipped:
    print(f"Skipped: {skipped}")

if results:
    top_hits, all_stats = collect_results(results, outdir=outdir, prefix="pesticides_cosine")
    print(f"\nSaved results to {outdir}")
    print(top_hits)