from pathlib import Path
from matchms import Spectrum
from matchms.importing import load_from_mgf
from matchms.similarity.FlashSimilarity import FlashSimilarity

from specreboot.preprocessing.filtering import general_cleaning
from specreboot.library.library_matching import (
    confidence_aware_match,
    collect_results,
    get_spectrum_id,
    summarize_top_annotation,
)

outdir = Path("/lustre/BIF/nobackup/charr003/projects/PostDoc/SpecReBoot_results/lib_match")
outdir.mkdir(parents=True, exist_ok=True)

# --- Load query spectra (clean if needed) ---
query_cleaned_path = outdir / "test_cleaned.mgf"
if query_cleaned_path.exists():
    query_spectra = list(load_from_mgf(str(query_cleaned_path)))
    print(f"Loaded {len(query_spectra)} pre-cleaned query spectra from {query_cleaned_path}")
else:
    query_spectra, _ = general_cleaning(
        load_from_mgf("/lustre/BIF/nobackup/charr003/projects/PostDoc/SpecReBoot_results/data/LibraryMatch_test.mgf"),
        file_name=str(query_cleaned_path),
    )
    print(f"Cleaned and saved {len(query_spectra)} query spectra to {query_cleaned_path}")

# --- Load library spectra (clean if needed) ---
library_cleaned_path = outdir / "library_cleaned.mgf"
if library_cleaned_path.exists():
    library_spectra = list(load_from_mgf(str(library_cleaned_path)))
    print(f"Loaded {len(library_spectra)} pre-cleaned library spectra from {library_cleaned_path}")
else:
    library_spectra, _ = general_cleaning(
        load_from_mgf("/lustre/BIF/nobackup/charr003/projects/PostDoc/SpecReBoot_results/data/MSn_COCONUT-featue-id.mgf"),
        file_name=str(library_cleaned_path),
    )
    print(f"Cleaned and saved {len(library_spectra)} library spectra to {library_cleaned_path}")

# --- Similarity metric ---
similarity_metric = FlashSimilarity(score_type="cosine", matching_mode="fragment", tolerance=0.01)

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
    top_hits, all_stats = collect_results(results, outdir=outdir, prefix="test_cosine")
    print(f"\nSaved results to {outdir}")
    print(top_hits)