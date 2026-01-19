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
    return_history: bool = False,
    track_bins: bool = False,
):
    """
    Compute bootstrapped spectral similarity and mutual-kNN edge support.

    Parameters
    ----------
    spectra_binned : list[matchms.Spectrum]
    global_bins : sequence of float
    B : int
        Number of bootstrap replicates.
    k : int
        Size of kNN neighbourhood (mutual kNN used for edge support).
    similarity_metric : matchms similarity object
    n_jobs : int
    seed : int
    return_history : bool, optional
        If True, also return per-bootstrap consensus similarity and edge
        support matrices.
    track_bins : bool, optional
        If True (and return_history is True), record sampled/missing m/z bins
        per bootstrap replicate.

    Returns
    -------
    df_mean_sim : pd.DataFrame
    df_edge_sup : pd.DataFrame
    history : dict, optional
        Only if return_history=True. Keys:
            - "mean_sim": list[np.ndarray]
            - "edge_sup": list[np.ndarray]
            - "sampled_bins": list[np.ndarray]   (if track_bins)
            - "missing_bins": list[np.ndarray]   (if track_bins)
    """

    rng = np.random.default_rng(seed)

    N = len(spectra_binned)
    P = len(global_bins)

    pair_sim_sum = defaultdict(float)
    pair_counts = defaultdict(int)
    edge_support = Counter()

    # -------------------------------------------------------------------------
    # Output labels + internal unique IDs
    # -------------------------------------------------------------------------
    scan_labels = []
    internal_ids = []
    id_to_index = dict()

    for idx, spec in enumerate(spectra_binned):
        scan_number = (
            spec.get("scans")
            or spec.metadata.get("scans")
            or spec.metadata.get("scan_number")
            or f"scan_{idx}"
        )
        scan_labels.append(str(scan_number))

        feature_id = spec.metadata.get("feature_id", f"feat_{idx}")
        internal_id = f"INTFID_{feature_id}_{idx}"
        internal_ids.append(internal_id)
        id_to_index[internal_id] = idx

    # -------------------------------------------------------------------------
    # Optional history containers
    # -------------------------------------------------------------------------
    hist_mean_sim = []
    hist_edge_sup = []
    hist_sampled_bins = []
    hist_missing_bins = []
    # -------------------------------------------------------------------------
    # Bootstrap iterations
    # -------------------------------------------------------------------------
    for b in range(B):

        # ------------------------------
        # 1. Sample bins (with replacement)
        # ------------------------------
        sampled_indices = rng.integers(0, P, size=P)
        sampled_bins_arr = np.asarray(global_bins)[sampled_indices]  # array, not set
        sampled_bins_arr = np.unique(sampled_bins_arr)               # optional speedup

        # ------------------------------
        # 2. Build bootstrap spectra
        # ------------------------------
        spectra_boot = []
        n_empty = 0

        for internal_id, spec in zip(internal_ids, spectra_binned):
            mz = spec.peaks.mz
            intens = spec.peaks.intensities

            mask = np.isin(mz, sampled_bins_arr)  # set is fine
            mz_kept = mz[mask]
            int_kept = intens[mask]

            if mz_kept.size == 0:
                n_empty += 1

            meta = {**spec.metadata, "internal_id": internal_id}

            spectra_boot.append(
                Spectrum(
                    mz=mz_kept.astype("float32"),
                    intensities=int_kept.astype("float32"),
                    metadata=meta,
                )
            )

        # ---- after building spectra_boot (ONCE per bootstrap) ----
        if n_empty > 0:
            for idx_s, s in enumerate(spectra_boot):
                if len(s.peaks.intensities) == 0:
                    spectra_boot[idx_s] = Spectrum(
                        mz=np.array([global_bins[0]], dtype="float32"),
                        intensities=np.array([0.0], dtype="float32"),
                        metadata=s.metadata,
                    )
            print(f"[bootstrap {b+1}/{B}] empty spectra this replicate: {n_empty}", flush=True)

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

        for item in scores:
            # matchms sometimes returns 3-tuple, sometimes ((ref,qry), score)
            if len(item) == 3:
                ref_spec, qry_spec, sim_val = item
            else:
                (ref_spec, qry_spec), sim_val = item

            i = id_to_index[ref_spec.metadata["internal_id"]]
            j = id_to_index[qry_spec.metadata["internal_id"]]

            if i == j:
                continue

            sim = float(sim_val[0])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

            key = tuple(sorted((internal_ids[i], internal_ids[j])))
            pair_sim_sum[key] += sim
            pair_counts[key] += 1

        # ------------------------------
        # 4. Mutual-kNN edges for this bootstrap
        # ------------------------------
        for i in range(N):
            row_sorted = np.argsort(sim_matrix[i])[::-1]
            row_sorted = row_sorted[row_sorted != i]
            knn_i = row_sorted[:k]

            for j in knn_i:
                row_j_sorted = np.argsort(sim_matrix[j])[::-1]
                row_j_sorted = row_j_sorted[row_j_sorted != j]
                knn_j = row_j_sorted[:k]

                if i < j and (i in knn_j):
                    key = tuple(sorted((internal_ids[i], internal_ids[j])))
                    edge_support[key] += 1

        # ------------------------------
        # 5. Optional: record history after this bootstrap
        # ------------------------------
        if return_history:
            # consensus similarity so far
            cur_mean = np.eye(N, dtype="float32")
            for (id_i, id_j), total in pair_sim_sum.items():
                idx_i = id_to_index[id_i]
                idx_j = id_to_index[id_j]
                cnt = pair_counts[(id_i, id_j)]
                cur_mean[idx_i, idx_j] = total / cnt
                cur_mean[idx_j, idx_i] = total / cnt

            # edge support so far (fraction of completed bootstraps)
            cur_edge = np.zeros((N, N), dtype="float32")
            denom = float(b + 1)
            for (id_i, id_j), cnt in edge_support.items():
                idx_i = id_to_index[id_i]
                idx_j = id_to_index[id_j]
                cur_edge[idx_i, idx_j] = cnt / denom
                cur_edge[idx_j, idx_i] = cnt / denom

            hist_mean_sim.append(cur_mean)
            hist_edge_sup.append(cur_edge)

            if track_bins:
                sampled_arr = np.unique(np.array(sampled_bins))
                missing_arr = np.setdiff1d(np.array(global_bins), sampled_arr)
                hist_sampled_bins.append(sampled_arr)
                hist_missing_bins.append(missing_arr)

        if (b + 1) % 5 == 0:
            print(f"[bootstrap {b+1}/{B}] done")

    # -------------------------------------------------------------------------
    # 6. Aggregation — map back to ORIGINAL scan labels
    # -------------------------------------------------------------------------
    mean_sim = np.eye(N, dtype="float32")

    for (id_i, id_j), total in pair_sim_sum.items():
        i = id_to_index[id_i]
        j = id_to_index[id_j]
        count = pair_counts[(id_i, id_j)]
        mean_sim[i, j] = total / count
        mean_sim[j, i] = total / count

    df_mean_sim = pd.DataFrame(mean_sim, index=scan_labels, columns=scan_labels)

    edge_mat = np.zeros((N, N), dtype="float32")

    for (id_i, id_j), cnt in edge_support.items():
        i = id_to_index[id_i]
        j = id_to_index[id_j]
        edge_mat[i, j] = cnt / B
        edge_mat[j, i] = cnt / B

    df_edge_sup = pd.DataFrame(edge_mat, index=scan_labels, columns=scan_labels)

    # -------------------------------------------------------------------------
    # 7. Return with or without history (backwards compatible)
    # -------------------------------------------------------------------------
    if not return_history:
        return df_mean_sim, df_edge_sup

    history = {
        "mean_sim": hist_mean_sim,
        "edge_sup": hist_edge_sup,
    }
    if track_bins:
        history["sampled_bins"] = hist_sampled_bins
        history["missing_bins"] = hist_missing_bins

    return df_mean_sim, df_edge_sup, history
