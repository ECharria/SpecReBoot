# Test data

## Library matching

| File | Contents |
|---|---|
| `libmatch_queries.mgf` | 20 annotated spectra drawn from GNPS-NP |
| `libmatch_library.mgf` | Those same 20, plus 180 distractors from the same source |

Every query has exactly one known correct answer in the library — its own
spectrum — so `tests/test_library_matching.py` can assert that retrieval
*works*, not merely that it runs. All 20 recover themselves at rank 1 with
`match_support` 1.0.

Sampled deterministically (seed 20260827) from
`GNPS-NP-feature-id_cleaned.mgf`, keeping only annotated spectra with at least
10 peaks. Together they are under half a megabyte, so the suite needs no
external data.

## Networking

| File | Contents |
|---|---|
| `test-exp-data-xylaria.mgf` | Experimental Xylaria spectra |
| `test-exp-data-xylaria.graphml` | Expected network output for the above |
