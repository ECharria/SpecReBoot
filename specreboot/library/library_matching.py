# specreboot/library/library_matching.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from matchms import Spectrum, calculate_scores

from specreboot.binning.binning import global_bins, bin_spectra


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass
class CandidateMatch:
    """Container for one original query-library match."""

    candidate_id: str
    original_score: float
    original_rank: int


@dataclass
class BootstrapCandidateStats:
    """Summary statistics for one candidate across bootstrap replicates."""

    candidate_id: str
    original_score: float
    original_rank: int
    match_support: float
    score_mean: float
    score_std: float
    top1_stability: float
    top3_stability: float
    top5_stability: float
    mean_rank: float
    distinct_top_hit_frequency: float


@dataclass
class ConfidenceAwareResult:
    """Full output for one query spectrum."""

    query_id: str
    top_candidate_id: str
    top_candidate_original_score: float
    candidate_stats: pd.DataFrame
    top_hit_frequencies: pd.DataFrame
    # Maps candidate_id → the actual matched library Spectrum object so that
    # reporting helpers can read spectrum.metadata directly without a second
    # pass through the full library.
    id_to_spectrum: Dict[str, Spectrum] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------

def get_spectrum_id(spectrum: Spectrum, fallback_prefix: str = "spec") -> str:
    """
    Return the best available identifier from a spectrum's metadata.

    Key priority is chosen so that GNPS library spectra (which carry both
    ``spectrum_id`` and ``feature_id``) are identified by their unique
    ``spectrum_id`` (e.g. ``CCMSLIB00000079350``) rather than the generic
    ``feature_id`` field (which is often ``'0'`` for every spectrum in a GNPS
    MGF export and is therefore not unique).  Query spectra produced by
    feature-finding tools typically lack a ``spectrum_id``, so ``feature_id``
    and ``scans`` serve as fallbacks for those.
    """
    md = spectrum.metadata or {}
    for key in ("spectrum_id", "feature_id", "id", "scans"):
        val = md.get(key)
        if val is not None and str(val) not in ("", "0", "None"):
            return str(val)
    return f"{fallback_prefix}_unknown"


def _unpack_score(sim_val: object) -> float:
    """
    Unpack a similarity value returned by ``calculate_scores``.

    matchms scorers differ in what they return:
    - ``FlashSimilarity`` returns a structured numpy scalar with a ``"score"``
      field alongside ``"n_matches"``; indexing by position (``sim_val[0]``)
      returns 0 because position 0 is ``n_matches``, not the score.
    - ``CosineGreedy`` / ``ModifiedCosine`` return a structured scalar with
      ``("score", "n_matches")`` where ``sim_val[0]`` happens to be the score,
      but named access is safer and consistent.
    - ``Spec2Vec`` / ``MS2DeepScore`` return a plain float-like scalar.

    Named field access is tried first; position-based and plain float are used
    as fallbacks so all scorers are handled without separate code paths.
    """
    try:
        return float(sim_val["score"])
    except (TypeError, KeyError, ValueError):
        try:
            return float(sim_val[0])
        except (TypeError, IndexError):
            return float(sim_val)


def score_candidates(
    query_spectrum: Spectrum,
    library_spectra: Sequence[Spectrum],
    similarity_metric: object,
) -> List[CandidateMatch]:
    """
    Score one query against all library spectra using ``calculate_scores``
    and return a ranked list of ``CandidateMatch`` objects.

    Parameters
    ----------
    query_spectrum:
        The query spectrum.
    library_spectra:
        All library spectra to score against.
    similarity_metric:
        Any matchms-compatible scorer (``FlashSimilarity``, ``CosineGreedy``,
        ``ModifiedCosine``, ``Spec2Vec``, ``MS2DeepScore``, …) passed
        directly — no wrapping needed.
    """
    scores = calculate_scores(
        references=list(library_spectra),
        queries=[query_spectrum],
        similarity_function=similarity_metric,
        is_symmetric=False,
    )

    matches = []
    for item in scores:
        if len(item) == 3:
            ref_spec, _qry_spec, sim_val = item
        else:
            (ref_spec, _qry_spec), sim_val = item

        candidate_id = get_spectrum_id(ref_spec, fallback_prefix="lib")
        matches.append((candidate_id, _unpack_score(sim_val)))

    matches.sort(key=lambda x: x[1], reverse=True)
    return [
        CandidateMatch(candidate_id=c_id, original_score=score, original_rank=i + 1)
        for i, (c_id, score) in enumerate(matches)
    ]


def restrict_library_to_top_n(
    ranked_matches: Sequence[CandidateMatch],
    library_spectra: Sequence[Spectrum],
    top_n: int,
) -> tuple[list[CandidateMatch], list[Spectrum]]:
    """Keep top-N matches and the corresponding library spectra."""
    top_matches = list(ranked_matches[:top_n])
    top_ids = {m.candidate_id for m in top_matches}
    restricted_library = [
        lib for lib in library_spectra
        if get_spectrum_id(lib, fallback_prefix="lib") in top_ids
    ]
    return top_matches, restricted_library


def compute_rank_stability_flags(
    ranked_candidate_ids: list[str],
    candidate_id: str,
) -> tuple[int, int, int, int]:
    """Return top-1/top-3/top-5 indicators and the observed rank."""
    if candidate_id not in ranked_candidate_ids:
        return 0, 0, 0, len(ranked_candidate_ids) + 1
    rank = ranked_candidate_ids.index(candidate_id) + 1
    return int(rank <= 1), int(rank <= 3), int(rank <= 5), rank


# -----------------------------------------------------------------------------
# Single bootstrap replicate
# -----------------------------------------------------------------------------

def _run_single_library_bootstrap(
    b: int,
    query_binned: Spectrum,
    library_binned: list[Spectrum],
    candidate_ids: list[str],
    global_bins_arr: np.ndarray,
    similarity_metric: object,
    seed: int,
    score_threshold: float,
) -> dict:
    """
    Run one bootstrap replicate for library matching.

    The bin-resampling kernel is identical to ``_run_single_bootstrap`` in
    ``bootstrapping.py``: the global bin array is resampled with replacement
    via ``rng.integers``, the unique sampled bins are retained, and each
    spectrum is filtered to only those bins via ``np.isin``.  A dummy peak is
    injected whenever all of a spectrum's peaks are filtered out.

    The only difference from the network bootstrap is the scoring step:
    instead of an all-vs-all symmetric ``calculate_scores`` call followed by
    mutual-kNN edge support, this function scores one query against all
    candidates asymmetrically (``is_symmetric=False``).

    Parameters
    ----------
    b:
        Replicate index; per-replicate seed is ``seed + b``.
    query_binned:
        Query spectrum already binned by ``bin_spectra``.
    library_binned:
        Top-N library spectra already binned by ``bin_spectra``, in the same
        order as ``candidate_ids``.
    candidate_ids:
        Ordered candidate identifiers matching ``library_binned``.
    global_bins_arr:
        The shared global bin array from ``global_bins`` as a numpy array.
    similarity_metric:
        Any matchms-compatible scorer passed directly to ``calculate_scores``.
    seed:
        Base random seed.
    score_threshold:
        Minimum score for a hit to count as supported.

    Returns
    -------
    dict with keys ``b``, ``boot_scores``, ``top_hit``, ``supported``,
    ``top1``, ``top3``, ``top5``, ``rank``.
    """
    rng = np.random.default_rng(seed + b)
    P = len(global_bins_arr)

    # --- Identical bin-resampling kernel as in _run_single_bootstrap ---
    sampled_indices = rng.integers(0, P, size=P)
    sampled_bins = np.unique(global_bins_arr[sampled_indices])

    def _filter_spectrum(spec: Spectrum, idx: int) -> Spectrum:
        mz = spec.peaks.mz
        intens = spec.peaks.intensities
        mask = np.isin(mz, sampled_bins)
        mz_kept = mz[mask]
        int_kept = intens[mask]

        # Inject dummy peak if all peaks were filtered out.
        if mz_kept.size == 0:
            dummy_mz = float(global_bins_arr.max() + 1000.0 + idx)
            mz_kept = np.array([dummy_mz], dtype="float32")
            int_kept = np.array([1.0], dtype="float32")

        return Spectrum(
            mz=mz_kept.astype("float32"),
            intensities=int_kept.astype("float32"),
            metadata=dict(spec.metadata),
        )

    boot_query = _filter_spectrum(query_binned, idx=0)
    boot_library = [_filter_spectrum(lib, idx=i + 1) for i, lib in enumerate(library_binned)]

    # --- Asymmetric scoring: one query vs. all candidates ---
    scores_obj = calculate_scores(
        references=boot_library,
        queries=[boot_query],
        similarity_function=similarity_metric,
        is_symmetric=False,
    )

    id_to_score: dict[str, float] = {}
    for item in scores_obj:
        if len(item) == 3:
            ref_spec, _qry, sim_val = item
        else:
            (ref_spec, _qry), sim_val = item
        cid = get_spectrum_id(ref_spec, fallback_prefix="lib")
        id_to_score[cid] = _unpack_score(sim_val)

    boot_scores = sorted(
        [(cid, id_to_score.get(cid, 0.0)) for cid in candidate_ids],
        key=lambda x: x[1],
        reverse=True,
    )
    ranked_ids = [cid for cid, _ in boot_scores]
    top_hit = ranked_ids[0]

    supported, top1_rec, top3_rec, top5_rec, rank_rec = {}, {}, {}, {}, {}
    for cid, score in boot_scores:
        top1, top3, top5, rank = compute_rank_stability_flags(ranked_ids, cid)
        supported[cid] = int(score >= score_threshold)
        top1_rec[cid] = top1
        top3_rec[cid] = top3
        top5_rec[cid] = top5
        rank_rec[cid] = rank

    return {
        "b": b,
        "boot_scores": boot_scores,
        "top_hit": top_hit,
        "supported": supported,
        "top1": top1_rec,
        "top3": top3_rec,
        "top5": top5_rec,
        "rank": rank_rec,
    }


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def confidence_aware_match(
    query_spectrum: Spectrum,
    library_spectra: Sequence[Spectrum],
    similarity_metric: object,
    B: int = 100,
    top_n: int = 100,
    score_threshold: float = 0.7,
    decimals: int = 2,
    seed: int = 42,
) -> ConfidenceAwareResult:
    """
    Confidence-aware library matching with SpecReBoot-style bootstrapping.

    Follows the same conventions as the rest of the SpecReBoot codebase:
    ``decimals`` controls the bin grid (matching ``--decimals`` in the CLI),
    binning is done via ``global_bins`` and ``bin_spectra`` from
    ``specreboot.binning.binning``, and the bin-resampling kernel in each
    bootstrap replicate is identical to ``_run_single_bootstrap`` in
    ``specreboot.bootstrapping.bootstrapping``.

    The global bin space is built from the query spectrum only, ensuring
    bootstrap replicates resample over bins that are actually informative for
    this query rather than over the full combined m/z space of all top-N
    library candidates (which would be dominated by library-only peaks).

    Workflow
    --------
    1. Score the original query against the full library with
       ``calculate_scores``.
    2. Restrict to the top-N candidates; build an ``id_to_spectrum`` mapping
       for direct metadata access in reporting.
    3. Build global bins from the query spectrum only.
    4. Bin the query and each library candidate with ``bin_spectra``.
    5. Run ``B`` bootstrap replicates via ``_run_single_library_bootstrap``.
    6. Summarise per-candidate support, score robustness, and rank stability.

    Parameters
    ----------
    query_spectrum:
        A matchms ``Spectrum`` object.
    library_spectra:
        All library spectra to search against.
    similarity_metric:
        Any matchms-compatible scorer passed directly — ``FlashSimilarity``,
        ``CosineGreedy``, ``ModifiedCosine``, ``Spec2Vec``,
        ``MS2DeepScore``, etc.  No wrapping needed.
    B:
        Number of bootstrap replicates (default 100).
    top_n:
        Number of top candidates to include in the bootstrap phase.
    score_threshold:
        Minimum bootstrap score for a hit to count as supported.
    decimals:
        Decimal places for m/z binning — mirrors ``--decimals`` in the CLI
        (default 2, i.e. 0.01 Da bins).
    seed:
        Base random seed; replicate ``b`` uses ``seed + b``.

    Returns
    -------
    ConfidenceAwareResult
    """
    query_id = get_spectrum_id(query_spectrum, fallback_prefix="query")

    if query_spectrum.peaks.mz.size == 0:
        raise ValueError(
            f"Query spectrum {query_id} has no peaks after cleaning. "
            "Check preprocessing output."
        )

    # Step 1–2: initial full-library search, restrict to top-N candidates
    ranked_matches = score_candidates(query_spectrum, library_spectra, similarity_metric)
    top_matches, restricted_library = restrict_library_to_top_n(
        ranked_matches, library_spectra, top_n
    )

    if not top_matches:
        raise ValueError(
            f"No library candidates were recovered for query {query_id}. "
            "The spectrum may have no peaks in common with the library."
        )

    top_candidate = top_matches[0]
    candidate_ids = [m.candidate_id for m in top_matches]

    # Build id → Spectrum mapping for direct metadata access in reporting.
    # This avoids any secondary lookup through the full library later.
    id_to_spectrum: Dict[str, Spectrum] = {
        get_spectrum_id(s, fallback_prefix="lib"): s
        for s in restricted_library
    }

    # Step 3: build global bins from the query only — ensures bootstrap
    # replicates resample over the query's actual peaks rather than the
    # full combined m/z space of all top-N candidates.
    bins = global_bins([query_spectrum], decimals)
    global_bins_arr = np.asarray(bins, dtype="float32")

    # Step 4: bin spectra, then normalise all m/z arrays to the same float32
    # representation so that np.isin() in the bootstrap works correctly.
    # Without this, global_bins() and bin_spectra() can produce values that
    # look identical when printed (e.g. 43.02) but differ at the binary level
    # due to float32/float64 rounding, causing np.isin to return all-False
    # masks and collapsing every bootstrap score to zero.
    def _normalise_mz(arr: np.ndarray) -> np.ndarray:
        return np.round(arr.astype("float64"), decimals).astype("float32")

    global_bins_arr = _normalise_mz(global_bins_arr)

    raw_query_binned = bin_spectra([query_spectrum], decimals)[0]
    query_binned = Spectrum(
        mz=_normalise_mz(raw_query_binned.peaks.mz),
        intensities=raw_query_binned.peaks.intensities,
        metadata=dict(raw_query_binned.metadata),
    )
    library_binned = [
        Spectrum(
            mz=_normalise_mz(s.peaks.mz),
            intensities=s.peaks.intensities,
            metadata=dict(s.metadata),
        )
        for s in bin_spectra(list(restricted_library), decimals)
    ]

    # Step 5: bootstrap replicates
    score_records:   dict[str, list[float]] = {cid: [] for cid in candidate_ids}
    support_records: dict[str, list[int]]   = {cid: [] for cid in candidate_ids}
    top1_records:    dict[str, list[int]]   = {cid: [] for cid in candidate_ids}
    top3_records:    dict[str, list[int]]   = {cid: [] for cid in candidate_ids}
    top5_records:    dict[str, list[int]]   = {cid: [] for cid in candidate_ids}
    rank_records:    dict[str, list[int]]   = {cid: [] for cid in candidate_ids}
    bootstrap_top_hits: list[str] = []

    for b in range(B):
        res = _run_single_library_bootstrap(
            b=b,
            query_binned=query_binned,
            library_binned=library_binned,
            candidate_ids=candidate_ids,
            global_bins_arr=global_bins_arr,
            similarity_metric=similarity_metric,
            seed=seed,
            score_threshold=score_threshold,
        )

        bootstrap_top_hits.append(res["top_hit"])

        for cid, score in res["boot_scores"]:
            score_records[cid].append(score)
            support_records[cid].append(res["supported"][cid])
            top1_records[cid].append(res["top1"][cid])
            top3_records[cid].append(res["top3"][cid])
            top5_records[cid].append(res["top5"][cid])
            rank_records[cid].append(res["rank"][cid])

    # Step 6: summarise
    top_hit_counts = pd.Series(bootstrap_top_hits).value_counts(normalize=True)

    stats_rows = []
    for match in top_matches:
        cid = match.candidate_id
        scores  = np.asarray(score_records[cid],   dtype=float)
        support = np.asarray(support_records[cid], dtype=float)
        top1s   = np.asarray(top1_records[cid],    dtype=float)
        top3s   = np.asarray(top3_records[cid],    dtype=float)
        top5s   = np.asarray(top5_records[cid],    dtype=float)
        ranks   = np.asarray(rank_records[cid],    dtype=float)

        stats_rows.append(BootstrapCandidateStats(
            candidate_id=cid,
            original_score=match.original_score,
            original_rank=match.original_rank,
            match_support=float(support.mean()),
            score_mean=float(scores.mean()),
            score_std=float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            top1_stability=float(top1s.mean()),
            top3_stability=float(top3s.mean()),
            top5_stability=float(top5s.mean()),
            mean_rank=float(ranks.mean()),
            distinct_top_hit_frequency=float(top_hit_counts.get(cid, 0.0)),
        ))

    candidate_stats = pd.DataFrame([vars(row) for row in stats_rows])
    candidate_stats = candidate_stats.sort_values(
        by=["original_rank", "top1_stability", "match_support"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    top_hit_frequencies = (
        pd.Series(bootstrap_top_hits)
        .value_counts()
        .rename_axis("candidate_id")
        .reset_index(name="count")
    )
    top_hit_frequencies["frequency"] = top_hit_frequencies["count"] / B

    return ConfidenceAwareResult(
        query_id=query_id,
        top_candidate_id=top_candidate.candidate_id,
        top_candidate_original_score=top_candidate.original_score,
        candidate_stats=candidate_stats,
        top_hit_frequencies=top_hit_frequencies,
        id_to_spectrum=id_to_spectrum,
    )


# -----------------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------------

def summarize_top_annotation(
    result: ConfidenceAwareResult,
    top_n: int = 5,
) -> str:
    """
    Return a human-readable summary of the top-N annotations for one query.

    Lists up to ``top_n`` candidates (default 5, matching the dreaMS
    convention) in rank order with their original score, key bootstrap
    metrics, compound name, and SMILES read directly from
    ``result.id_to_spectrum[cid].metadata``.

    Parameters
    ----------
    result:
        A ``ConfidenceAwareResult`` returned by ``confidence_aware_match``.
    top_n:
        Maximum number of candidates to include (default 5).
    """
    rows = result.candidate_stats.head(top_n)
    lines = [f"Query {result.query_id} — top-{min(top_n, len(rows))} annotations:"]

    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        cid  = row["candidate_id"]
        spec = result.id_to_spectrum.get(cid)
        md   = spec.metadata if spec else {}
        name   = md.get("compound_name") or md.get("name") or cid
        smiles = md.get("smiles")

        line = (
            f"  [{i}] {name} (id={cid})"
            + (f" | SMILES: {smiles}" if smiles else "")
            + f"\n      score={row['original_score']:.3f}"
            f"  support={row['match_support']:.3f}"
            f"  top1={row['top1_stability']:.3f}"
            f"  top5={row['top5_stability']:.3f}"
            f"  robustness={row['score_mean']:.3f}±{row['score_std']:.3f}"
        )
        lines.append(line)

    return "\n".join(lines)


def _build_top_n_df(
    result: ConfidenceAwareResult,
    top_n: int,
) -> pd.DataFrame:
    """
    Build a top-N DataFrame for one query enriched with library metadata
    read directly from ``result.id_to_spectrum[cid].metadata``.
    One row per candidate, ranked 1 to N.
    """
    rows = []
    for rank, (_, stats_row) in enumerate(
        result.candidate_stats.head(top_n).iterrows(), start=1
    ):
        cid  = stats_row["candidate_id"]
        spec = result.id_to_spectrum.get(cid)
        md   = spec.metadata if spec else {}

        rows.append({
            "query_id":          result.query_id,
            "rank":              rank,
            "candidate_id":      cid,
            # metadata read straight from spectrum.metadata
            "name":              md.get("compound_name") or md.get("name"),
            "smiles":            md.get("smiles"),
            "inchikey":          md.get("inchikey"),
            "inchi":             md.get("inchi"),
            "formula":           md.get("formula"),
            "precursor_mz":      md.get("precursor_mz"),
            "adduct":            md.get("adduct"),
            "instrument_type":   md.get("instrument_type"),
            "collision_energy":  md.get("collision_energy"),
            # bootstrap metrics
            "original_score":             stats_row["original_score"],
            "original_rank":              stats_row["original_rank"],
            "match_support":              stats_row["match_support"],
            "score_mean":                 stats_row["score_mean"],
            "score_std":                  stats_row["score_std"],
            "top1_stability":             stats_row["top1_stability"],
            "top3_stability":             stats_row["top3_stability"],
            "top5_stability":             stats_row["top5_stability"],
            "mean_rank":                  stats_row["mean_rank"],
            "distinct_top_hit_frequency": stats_row["distinct_top_hit_frequency"],
        })

    return pd.DataFrame(rows)


def save_results(
    result: ConfidenceAwareResult,
    outdir: "str | Path",
    prefix: str = "lib_match",
    top_n: int = 5,
) -> None:
    """
    Save a ``ConfidenceAwareResult`` to CSV files.

    Writes two files:

    - ``<prefix>_top_hits_<query_id>.csv`` — top-N candidates with bootstrap
      statistics and library metadata read directly from matched spectrum
      objects.
    - ``<prefix>_top_hit_frequencies_<query_id>.csv`` — bootstrap top-hit
      frequency table.

    Parameters
    ----------
    result:
        A ``ConfidenceAwareResult`` returned by ``confidence_aware_match``.
    outdir:
        Output directory.  Created if it does not exist.
    prefix:
        Filename prefix.
    top_n:
        Number of top candidates to save (default 5).
    """
    from pathlib import Path

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    qid = result.query_id
    _build_top_n_df(result, top_n).to_csv(
        outdir / f"{prefix}_top_hits_{qid}.csv", index=False
    )
    result.top_hit_frequencies.to_csv(
        outdir / f"{prefix}_top_hit_frequencies_{qid}.csv", index=False
    )


def collect_results(
    results: "list[ConfidenceAwareResult]",
    outdir: "str | Path",
    prefix: str = "lib_match",
    top_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate results for all queries and save to CSV.

    Produces two combined files:

    - ``<prefix>_all_top_hits.csv`` — up to ``top_n`` rows per query (one per
      candidate), with bootstrap statistics and library metadata read directly
      from the matched spectrum objects.  Filter to top-1 with
      ``df[df["rank"] == 1]``.
    - ``<prefix>_all_candidate_stats.csv`` — full per-candidate bootstrap
      statistics for every query concatenated into a single table.

    Parameters
    ----------
    results:
        List of ``ConfidenceAwareResult`` objects, one per query spectrum.
    outdir:
        Output directory.  Created if it does not exist.
    prefix:
        Filename prefix.
    top_n:
        Maximum number of candidates to report per query (default 5).

    Returns
    -------
    tuple of (all_top_hits, all_candidate_stats) DataFrames, also written
    to CSV.

    Examples
    --------
    ::

        results = []
        for query in query_spectra:
            try:
                results.append(confidence_aware_match(query, library, similarity_metric))
            except ValueError as e:
                print(f"Skipping {get_spectrum_id(query)}: {e}")

        top_hits, all_stats = collect_results(results, outdir="results/", prefix="my_run")
        # Filter to top-1 only:
        top1 = top_hits[top_hits["rank"] == 1]
    """
    from pathlib import Path

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_top_hits = pd.concat(
        [_build_top_n_df(r, top_n) for r in results], ignore_index=True
    )
    all_top_hits.to_csv(outdir / f"{prefix}_all_top_hits.csv", index=False)

    stats_frames = []
    for r in results:
        df = r.candidate_stats.copy()
        df.insert(0, "query_id", r.query_id)
        stats_frames.append(df)

    all_candidate_stats = pd.concat(stats_frames, ignore_index=True)
    all_candidate_stats.to_csv(outdir / f"{prefix}_all_candidate_stats.csv", index=False)

    return all_top_hits, all_candidate_stats