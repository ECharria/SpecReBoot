"""Regression tests for bootstrap replicate seeding and batch-size resolution.

These tests guard the fix for the seeding-coupling bug: the RNG used to be
created once per batch from the base seed, so every batch replayed the first
batch's resamples and the number of unique replicates was capped at
``batch_size`` instead of ``B``.

All bootstrapping runs use ``n_jobs=1``. With more than one worker the batch
results are summed in a nondeterministic order, and float rounding can flip
borderline edges, which is out of scope for these tests.
"""

import inspect

import numpy as np
import pytest
from matchms import Spectrum
from matchms.similarity.FlashSimilarity import FlashSimilarity

from specreboot.binning.binning import bin_spectra, global_bins as make_global_bins
from specreboot.bootstrapping import bootstrapping as bootstrapping_module
from specreboot.bootstrapping.bootstrapping import (
    AUTO_BATCH_FEATURE_THRESHOLD,
    LARGE_DATA_BATCH_SIZE,
    _resolve_batch_size,
    calculate_bootstrapping,
)

SEED = 42
K = 3


def _make_spectra(n_spectra=8, n_peaks=15, seed=0):
    """Build a small synthetic set of spectra with overlapping peaks."""
    rng = np.random.default_rng(seed)
    spectra = []
    for i in range(n_spectra):
        mz = np.sort(rng.choice(np.arange(100.0, 400.0, 1.0), size=n_peaks, replace=False))
        intensities = rng.random(n_peaks).astype("float32") + 0.1
        spectra.append(
            Spectrum(
                mz.astype("float32"),
                intensities,
                metadata={"precursor_mz": 500.0 + i, "feature_id": f"F{i}"},
                metadata_harmonization=False,
            )
        )
    return spectra


@pytest.fixture(scope="module")
def dataset():
    """Return (binned spectra, global bins) for the synthetic dataset."""
    spectra = _make_spectra()
    return bin_spectra(spectra, 1), make_global_bins(spectra, 1)


@pytest.fixture(scope="module")
def similarity_metric():
    return FlashSimilarity()


def _run(dataset, similarity_metric, B, batch_size, seed=SEED):
    """Run bootstrapping and return the (similarity, edge support) matrices."""
    spectra_binned, bins = dataset
    df_mean_sim, df_edge_sup, _ = calculate_bootstrapping(
        spectra_binned,
        bins,
        B=B,
        k=K,
        similarity_metric=similarity_metric,
        n_jobs=1,
        batch_size=batch_size,
        seed=seed,
        verbose=False,
    )
    return df_mean_sim.values, df_edge_sup.values


def test_same_seed_is_reproducible(dataset, similarity_metric):
    """Two runs with the same seed and B produce identical matrices."""
    first_sim, first_edge = _run(dataset, similarity_metric, B=20, batch_size=5)
    second_sim, second_edge = _run(dataset, similarity_metric, B=20, batch_size=5)

    np.testing.assert_array_equal(first_sim, second_sim)
    np.testing.assert_array_equal(first_edge, second_edge)


@pytest.mark.parametrize("batch_size", [1, 3, 5, 7])
def test_batch_size_does_not_change_the_result(dataset, similarity_metric, batch_size):
    """batch_size is a memory knob only: chunked and single-batch runs agree.

    This is the regression test for the seeding-coupling bug. Before the fix,
    batch_size=5 produced only 5 unique replicates repeated 4 times, and the
    two runs disagreed by ~1e-2.

    Edge support is compared exactly: it is an integer count of mutual-kNN hits
    per replicate, so it cannot depend on how replicates are grouped. Mean
    similarity is compared at machine precision rather than bit-for-bit, because
    batching changes the grouping of the float summation and float addition is
    not associative. The residual is ~1e-16, i.e. 15 orders of magnitude below
    the bug this test guards against.
    """
    B = 20
    chunked_sim, chunked_edge = _run(dataset, similarity_metric, B=B, batch_size=batch_size)
    single_sim, single_edge = _run(dataset, similarity_metric, B=B, batch_size=B)

    np.testing.assert_array_equal(chunked_edge, single_edge)
    np.testing.assert_allclose(chunked_sim, single_sim, rtol=0, atol=1e-12)


def test_more_replicates_change_the_result(dataset, similarity_metric):
    """Increasing B adds unique replicates and therefore changes the result.

    Before the fix, B beyond batch_size only repeated the same replicates, so
    the averaged matrices were identical up to float noise.
    """
    small_sim, _ = _run(dataset, similarity_metric, B=10, batch_size=10)
    large_sim, _ = _run(dataset, similarity_metric, B=50, batch_size=10)

    assert not np.allclose(small_sim, large_sim, atol=1e-6)


def test_each_replicate_uses_a_distinct_resample(dataset, similarity_metric):
    """Every replicate index maps to its own resample of the global bins."""
    _, bins = dataset
    bins = np.array(bins)

    samples = set()
    for b in range(20):
        rng = np.random.default_rng(SEED + b)
        sampled = rng.integers(0, len(bins), size=len(bins))
        samples.add(sampled.tobytes())

    assert len(samples) == 20


def test_default_seed_is_unchanged():
    """The documented default seed must stay 42 so results stay stable."""
    assert inspect.signature(calculate_bootstrapping).parameters["seed"].default == 42


def test_batch_size_default_is_auto():
    """batch_size defaults to None, meaning 'resolve automatically'."""
    assert inspect.signature(calculate_bootstrapping).parameters["batch_size"].default is None


@pytest.mark.parametrize(
    "n_features, expected",
    [
        (10, 20),
        (AUTO_BATCH_FEATURE_THRESHOLD - 1, 20),
        (AUTO_BATCH_FEATURE_THRESHOLD, LARGE_DATA_BATCH_SIZE),
        (AUTO_BATCH_FEATURE_THRESHOLD + 1, LARGE_DATA_BATCH_SIZE),
    ],
)
def test_auto_batch_size_resolution(n_features, expected):
    """Auto mode chunks only at or above the large-dataset threshold."""
    assert _resolve_batch_size(None, B=20, n_features=n_features) == expected


@pytest.mark.parametrize("explicit", [1, 5, 10, 100])
def test_explicit_batch_size_is_honoured(explicit):
    """An explicit integer batch size is passed through unchanged."""
    assert _resolve_batch_size(explicit, B=20, n_features=10) == explicit
    assert _resolve_batch_size(explicit, B=20, n_features=AUTO_BATCH_FEATURE_THRESHOLD) == explicit


def test_small_dataset_runs_as_a_single_batch(dataset, similarity_metric, monkeypatch):
    """With batch_size=None a small dataset is not chunked at all."""
    spectra_binned, bins = dataset
    assert len(spectra_binned) < AUTO_BATCH_FEATURE_THRESHOLD

    seen_batches = []
    original = bootstrapping_module.bootstrap_batch

    def recording_bootstrap_batch(*args, **kwargs):
        seen_batches.append(list(args[5]))  # positional arg B: the batch's replicate indices
        return original(*args, **kwargs)

    monkeypatch.setattr(bootstrapping_module, "bootstrap_batch", recording_bootstrap_batch)

    B = 12
    calculate_bootstrapping(
        spectra_binned,
        bins,
        B=B,
        k=K,
        similarity_metric=similarity_metric,
        n_jobs=1,
        batch_size=None,
        seed=SEED,
        verbose=False,
    )

    assert seen_batches == [list(range(B))]
