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
    """


    rng = np.random.default_rng(seed)

    N = len(spectra_binned)
    P = len(global_bins)

    pair_sim_sum = defaultdict(float)
    pair_counts = defaultdict(int)
    edge_support = Counter()

    # -------------------------------------------------------------------------
    # Create output scan labels + internal unique IDs
    # -------------------------------------------------------------------------
    scan_labels = []      # final labels (user sees these)
    internal_ids = []     # unique per spectrum (used internally)
    id_to_index = dict()

    for idx, spec in enumerate(spectra_binned):

        # ORIGINAL scan number (may be duplicated)
        scan_number = (
            spec.get("scans")
            or spec.metadata.get("scans")
            or spec.metadata.get("scan_number")
            or f"scan_{idx}"
        )
        scan_labels.append(str(scan_number))

        # INTERNAL unique ID based on feature_id
        feature_id = spec.metadata.get("feature_id", f"feat_{idx}")
        internal_id = f"INTFID_{feature_id}_{idx}"
        internal_ids.append(internal_id)

        id_to_index[internal_id] = idx

    # -------------------------------------------------------------------------
    # Bootstrap iterations
    # -------------------------------------------------------------------------
    for _ in range(B):

        # ------------------------------
        # 1. Sample bins
        # ------------------------------
        sampled_indices = rng.integers(0, P, size=P)
        sampled_bins = set(global_bins[i] for i in sampled_indices)

        # ------------------------------
        # 2. Build bootstrap spectra
        # ------------------------------
        spectra_boot = []
        for internal_id, spec in zip(internal_ids, spectra_binned):

            mz = spec.peaks.mz
            intens = spec.peaks.intensities

            mask = np.isin(mz, list(sampled_bins))
            mz_kept = mz[mask]
            int_kept = intens[mask]

            # Insert internal ID into metadata for matching
            meta = {**spec.metadata, "internal_id": internal_id}

            spectra_boot.append(
                Spectrum(
                    mz=mz_kept.astype("float32"),
                    intensities=int_kept.astype("float32"),
                    metadata=meta,
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

        for item in scores:
            if len(item) == 3:
                ref_spec, qry_spec, sim_val = item
            else:
                (ref_spec, qry_spec), sim_val = item

            # Identify by INTERNAL ID (never by scan)
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
        # 4. Mutual-kNN edges
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

    # -------------------------------------------------------------------------
    # 5. Aggregation — map back to ORIGINAL scan labels
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

    return df_mean_sim, df_edge_sup