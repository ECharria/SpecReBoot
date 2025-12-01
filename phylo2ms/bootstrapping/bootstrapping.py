import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from matchms import calculate_scores
from matchms import Spectrum
from joblib import parallel_backend


def calculate_boostrapping(
    spectra_binned,
    global_bins,
    B=100,
    k=5,
    similarity_metric=None,
    n_jobs=8,
    seed=42,
):
    """
    Compute bootstrapped spectral similarity and mutual-kNN edge support.

    Parameters
    ----------
    spectra_binned : list of Spectrum
        Spectra with peaks already binned (rounded m/z values).
    global_bins : list of float
        Global set of all possible peak bins from the dataset.
    B : int, optional (default=100)
        Number of bootstrap iterations.
    k : int, optional (default=5)
        k in mutual-kNN detection.
    similarity_metric : matchms.similarity object
        Similarity function, e.g., ModifiedCosine().
    n_jobs : int, optional (default=8)
        Number of parallel worker jobs.
    seed : int, optional (default=42)
        Random generator seed.

    Returns
    -------
    df_mean_sim : pd.DataFrame (N x N)
        Mean spectral similarity over bootstraps.
    df_edge_sup : pd.DataFrame (N x N)
        Edge support: fraction of bootstraps where the edge
        is mutual k-nearest-neighbors.
    """
    rng = np.random.default_rng(seed)

    N = len(spectra_binned)
    P = len(global_bins)

    # Track summed similarities and counts
    pair_sim_sum = defaultdict(float)
    pair_counts = defaultdict(int)

    # Track mutual-kNN edge support
    edge_support = Counter()

    # --- Resolve scan IDs ---
    scan_ids = []
    seen_ids = set()
    for idx, spec in enumerate(spectra_binned):
        sid = (
            spec.get("scans")
            or spec.metadata.get("scans")
            or spec.metadata.get("spectrum_id")
            or spec.metadata.get("title")
            or spec.metadata.get("feature_id")
            or f"spectrum_{idx}"
        )
        # Force uniqueness
        if sid in seen_ids:
            sid = f"{sid}_{idx}"
        scan_ids.append(sid)
        seen_ids.add(sid)
    scan_to_index = {sid: idx for idx, sid in enumerate(scan_ids)}
    # -------------------------------------------------------------------------
    # Bootstrap loop
    # -------------------------------------------------------------------------
    for _ in range(B):
        # ------------------------------
        # 1. Sample bins with replacement
        # ------------------------------
        sampled_indices = rng.integers(0, P, size=P)
        sampled_bins = set(global_bins[i] for i in sampled_indices)

        # ------------------------------
        # 2. Build bootstrapped spectra
        # ------------------------------
        spectra_boot = []
        for spec in spectra_binned:
            mz = spec.peaks.mz
            intens = spec.peaks.intensities

            mask = np.isin(mz, list(sampled_bins))
            mz_kept = mz[mask]
            int_kept = intens[mask]

            spectra_boot.append(
                Spectrum(
                    mz=mz_kept.astype("float32"),
                    intensities=int_kept.astype("float32"),
                    metadata=spec.metadata,
                )
            )

        # ------------------------------
        # 3. Compute similarity matrix
        # ------------------------------
        sim_matrix = np.zeros((N, N), dtype=float)

        with parallel_backend("loky", n_jobs=n_jobs):
            scores = calculate_scores(
                references=spectra_boot,
                queries=spectra_boot,
                similarity_function=similarity_metric,
                is_symmetric=True,
            )

        # Parse similarity results
        for item in scores:
            if len(item) == 3:
                ref_spec, qry_spec, sim_val = item
            else:
                (ref_spec, qry_spec), sim_val = item

            i = scan_to_index[ref_spec.get("scans")]
            j = scan_to_index[qry_spec.get("scans")]
            if i == j:
                continue

            sim = float(sim_val[0])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

            pair_sim_sum[(scan_ids[i], scan_ids[j])] += sim
            pair_counts[(scan_ids[i], scan_ids[j])] += 1

        # ------------------------------
        # 4. Mutual-kNN detection
        # ------------------------------
        for i in range(N):
            # i → k nearest neighbors
            row_sorted = np.argsort(sim_matrix[i])[::-1]
            row_sorted = row_sorted[row_sorted != i]
            knn_i = row_sorted[:k]

            for j in knn_i:
                # j → k nearest neighbors
                row_j_sorted = np.argsort(sim_matrix[j])[::-1]
                row_j_sorted = row_j_sorted[row_j_sorted != j]
                knn_j = row_j_sorted[:k]

                # Mutual neighbor check; prevent double-counting with i < j
                if i < j and (i in knn_j):
                    key = tuple(sorted((scan_ids[i], scan_ids[j])))
                    edge_support[key] += 1

    # -------------------------------------------------------------------------
    # Aggregate results
    # -------------------------------------------------------------------------
    # Mean similarity matrix
    mean_sim = np.eye(N, dtype="float32")

    for (sid_i, sid_j), total in pair_sim_sum.items():
        i = scan_to_index[sid_i]
        j = scan_to_index[sid_j]
        count = pair_counts[(sid_i, sid_j)]
        mean_sim[i, j] = total / count
        mean_sim[j, i] = total / count

    df_mean_sim = pd.DataFrame(mean_sim, index=scan_ids, columns=scan_ids)

    # Edge-support matrix
    edge_mat = np.zeros((N, N), dtype="float32")

    for (sid_i, sid_j), cnt in edge_support.items():
        i = scan_to_index[sid_i]
        j = scan_to_index[sid_j]
        edge_mat[i, j] = cnt / B
        edge_mat[j, i] = cnt / B

    df_edge_sup = pd.DataFrame(edge_mat, index=scan_ids, columns=scan_ids)

    return df_mean_sim, df_edge_sup


