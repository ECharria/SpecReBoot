"""Library matching against a small, self-contained dataset.

``tests/data/libmatch_queries.mgf`` holds 20 annotated spectra drawn from
GNPS-NP; ``tests/data/libmatch_library.mgf`` holds those same 20 plus 180
distractors from the same source. Every query therefore has exactly one known
correct answer in the library — its own spectrum — which is what makes these
assertions meaningful rather than merely smoke tests.

Both files are small enough to live in the repository (39 KB and 415 KB), so the
suite needs no external data.
"""
from pathlib import Path

import pytest
from matchms.filtering import normalize_intensities
from matchms.importing import load_from_mgf
from matchms.similarity import CosineGreedy, ModifiedCosine

from specreboot.library.library_matching import (
    confidence_aware_match,
    filter_by_precursor_mz,
    get_spectrum_id,
)

DATA = Path(__file__).parent / "data"


def _load(name):
    return [normalize_intensities(s) for s in load_from_mgf(str(DATA / name)) if s]


@pytest.fixture(scope="module")
def queries():
    return _load("libmatch_queries.mgf")


@pytest.fixture(scope="module")
def library():
    return _load("libmatch_library.mgf")


@pytest.fixture(scope="module")
def matched(queries, library):
    """One match per query, so the expensive part runs once for the module."""
    out = []
    for spectrum in queries:
        result = confidence_aware_match(
            query_spectrum=spectrum,
            library_spectra=library,
            similarity_metric=ModifiedCosine(),
            B=20, top_n=10, score_threshold=0.7, decimals=2, seed=42,
            precursor_mz_tolerance_da=0.02,
        )
        out.append((spectrum, result))
    return out


class TestDataset:
    def test_queries_are_present_in_the_library(self, queries, library):
        library_ids = {get_spectrum_id(s) for s in library}
        missing = [get_spectrum_id(q) for q in queries if get_spectrum_id(q) not in library_ids]
        assert not missing, f"queries with no true match in the library: {missing}"

    def test_every_query_is_annotated(self, queries):
        assert all(q.get("compound_name") for q in queries)

    def test_library_is_mostly_distractors(self, queries, library):
        assert len(library) > 5 * len(queries)


class TestRetrieval:
    def test_every_query_recovers_itself_as_the_top_hit(self, matched):
        """The strongest assertion available: a known answer per query."""
        missed = []
        for spectrum, result in matched:
            stats = result.candidate_stats
            if not len(stats) or str(stats.iloc[0]["candidate_id"]) != get_spectrum_id(spectrum):
                missed.append(get_spectrum_id(spectrum))
        assert not missed, f"{len(missed)} queries did not rank their own spectrum first: {missed}"

    def test_the_true_match_is_fully_supported(self, matched):
        """A self-match survives every replicate, so support should be 1.0."""
        for spectrum, result in matched:
            top = result.candidate_stats.iloc[0]
            assert top["match_support"] == pytest.approx(1.0), get_spectrum_id(spectrum)

    def test_reports_every_documented_metric(self, matched):
        expected = {
            "candidate_id", "original_score", "original_rank", "match_support",
            "score_mean", "score_std", "top1_stability", "top3_stability",
            "top5_stability", "mean_rank",
        }
        _, result = matched[0]
        assert expected <= set(result.candidate_stats.columns)

    def test_support_never_leaves_the_unit_interval(self, matched):
        for _, result in matched:
            support = result.candidate_stats["match_support"]
            assert support.between(0.0, 1.0).all()

    def test_results_are_deterministic_for_a_fixed_seed(self, queries, library):
        kwargs = dict(
            library_spectra=library, similarity_metric=ModifiedCosine(),
            B=10, top_n=5, score_threshold=0.7, decimals=2, seed=7,
            precursor_mz_tolerance_da=0.02,
        )
        first = confidence_aware_match(query_spectrum=queries[0], **kwargs)
        second = confidence_aware_match(query_spectrum=queries[0], **kwargs)
        assert first.candidate_stats["match_support"].tolist() == \
               second.candidate_stats["match_support"].tolist()


class TestPrecursorTolerance:
    """The tolerance is a user-facing knob, so its effect is pinned here."""

    def test_a_tighter_window_keeps_fewer_candidates(self, queries, library):
        query = queries[0]
        counts = [
            len(filter_by_precursor_mz(query, library, tolerance_da=tol))
            for tol in (0.005, 0.05, 1.0, 50.0)
        ]
        assert counts == sorted(counts), counts
        assert counts[0] < counts[-1]

    def test_the_true_match_survives_a_tight_window(self, queries, library):
        """Filtering must never discard the answer it is meant to find."""
        for query in queries[:5]:
            kept = filter_by_precursor_mz(query, library, tolerance_da=0.02)
            assert get_spectrum_id(query) in {get_spectrum_id(s) for s in kept}

    def test_widening_the_window_does_not_change_the_top_hit(self, queries, library):
        """More distractors admitted, same answer — the point of the ranking."""
        query = queries[0]
        tops = []
        for tol in (0.02, 5.0):
            result = confidence_aware_match(
                query_spectrum=query, library_spectra=library,
                similarity_metric=ModifiedCosine(), B=10, top_n=10,
                score_threshold=0.7, decimals=2, seed=42,
                precursor_mz_tolerance_da=tol,
            )
            tops.append(str(result.candidate_stats.iloc[0]["candidate_id"]))
        assert tops[0] == tops[1] == get_spectrum_id(query)


class TestMetrics:
    @pytest.mark.parametrize("metric", [CosineGreedy(), ModifiedCosine()])
    def test_the_cosine_family_recovers_the_true_match(self, queries, library, metric):
        query = queries[0]
        result = confidence_aware_match(
            query_spectrum=query, library_spectra=library, similarity_metric=metric,
            B=10, top_n=5, score_threshold=0.7, decimals=2, seed=42,
            precursor_mz_tolerance_da=0.02,
        )
        assert str(result.candidate_stats.iloc[0]["candidate_id"]) == get_spectrum_id(query)
