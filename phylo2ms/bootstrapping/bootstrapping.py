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
    label_mode: str = "feature",      # "feature" | "scan" | "internal"
    return_label_map: bool = True,
):
    """
    Bootstrapped mean similarity + mutual-kNN edge support.

    Returns
    -------
    If return_history is False:
        df_mean_sim, df_edge_sup                 (and label_map if return_label_map True)
    If return_history is True:
        df_mean_sim, df_edge_sup, history        (history may include label_map)
    """

    # --- guardrails for the exact error you hit ---
    if global_bins is None or not hasattr(global_bins, "__len__"):
        raise TypeError(
            "global_bins must be an array/list of bin m/z values. "
            "Did you accidentally pass the global_bins *function* instead of the computed bins array?"
        )

    def make_unique(labels):
        counts = Counter(labels)
        seen = Counter()
        out = []
        for lab in labels:
            lab = str(lab)
            if counts[lab] == 1:
                out.append(lab)
            else:
                seen[lab] += 1
                out.append(f"{lab}__{seen[lab]}")
        return out

    rng = np.random.default_rng(seed)

    N = len(spectra_binned)
    global_bins_arr = np.asarray(global_bins)
    P = len(global_bins_arr)

    pair_sim_sum = defaultdict(float)
    pair_counts = defaultdict(int)
    edge_support = Counter()

    # --- labels + unique internal IDs for mapping scores back to indices ---
    scan_labels = []
    feature_labels = []
    internal_ids = []
    id_to_index = {}

    for idx, spec in enumerate(spectra_binned):
        md = spec.metadata

        scan_number = (
            getattr(spec, "get", lambda x: None)("scans")
            or md.get("scans") or md.get("SCANS")
            or md.get("scan_number") or md.get("SCAN_NUMBER")
            or f"scan_{idx}"
        )
        scan_labels.append(str(scan_number))

        feature_id = md.get("feature_id", md.get("FEATURE_ID", f"feat_{idx}"))
        feature_labels.append(str(feature_id))

        internal_id = f"INTFID_{feature_id}_{idx}"
        internal_ids.append(internal_id)
        id_to_index[internal_id] = idx

    # make labels safe for DataFrames/graphs
    scan_labels_u = make_unique(scan_labels)
    feature_labels_u = make_unique(feature_labels)
    internal_ids_u = make_unique(internal_ids)

    lm = (label_mode or "feature").lower()
    if lm == "scan":
        out_labels = scan_labels_u
    elif lm == "internal":
        out_labels = internal_ids_u
    else:
        out_labels = feature_labels_u  # default

    label_map = pd.DataFrame({
        "out_label": out_labels,
        "scan": scan_labels,
        "scan_unique": scan_labels_u,
        "feature_id": feature_labels,
        "feature_unique": feature_labels_u,
        "internal_id": internal_ids,
    })

    # --- optional history ---
    hist_mean_sim = []
    hist_edge_sup = []
    hist_sampled_bins = []
    hist_missing_bins = []

    # --- bootstraps ---
    for b in range(B):
        sampled_indices = rng.integers(0, P, size=P)
        sampled_bins_arr = np.unique(global_bins_arr[sampled_indices])  # IMPORTANT: array, not set

        spectra_boot = []
        n_empty = 0

        for internal_id, spec in zip(internal_ids, spectra_binned):
            mz = spec.peaks.mz
            intens = spec.peaks.intensities

            mask = np.isin(mz, sampled_bins_arr)
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

        # dummy peak so matchms/similarity functions don't crash on empty spectra
        if n_empty > 0:
            for i_s, s in enumerate(spectra_boot):
                if len(s.peaks.intensities) == 0:
                    spectra_boot[i_s] = Spectrum(
                        mz=np.array([global_bins_arr[0]], dtype="float32"),
                        intensities=np.array([0.0], dtype="float32"),
                        metadata=s.metadata,
                    )
            print(f"[bootstrap {b+1}/{B}] empty spectra this replicate: {n_empty}", flush=True)

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

        # mutual-kNN edges
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

        # history
        if return_history:
            cur_mean = np.eye(N, dtype="float32")
            for (id_i, id_j), total in pair_sim_sum.items():
                ii = id_to_index[id_i]
                jj = id_to_index[id_j]
                cnt = pair_counts[(id_i, id_j)]
                cur_mean[ii, jj] = total / cnt
                cur_mean[jj, ii] = total / cnt

            cur_edge = np.zeros((N, N), dtype="float32")
            denom = float(b + 1)
            for (id_i, id_j), cnt in edge_support.items():
                ii = id_to_index[id_i]
                jj = id_to_index[id_j]
                cur_edge[ii, jj] = cnt / denom
                cur_edge[jj, ii] = cnt / denom

            hist_mean_sim.append(cur_mean)
            hist_edge_sup.append(cur_edge)

            if track_bins:
                sampled_arr = np.unique(sampled_bins_arr)
                missing_arr = np.setdiff1d(global_bins_arr, sampled_arr)
                hist_sampled_bins.append(sampled_arr)
                hist_missing_bins.append(missing_arr)

        if (b + 1) % 5 == 0:
            print(f"[bootstrap {b+1}/{B}] done")

    # aggregate
    mean_sim = np.eye(N, dtype="float32")
    for (id_i, id_j), total in pair_sim_sum.items():
        i = id_to_index[id_i]
        j = id_to_index[id_j]
        cnt = pair_counts[(id_i, id_j)]
        mean_sim[i, j] = total / cnt
        mean_sim[j, i] = total / cnt

    edge_mat = np.zeros((N, N), dtype="float32")
    for (id_i, id_j), cnt in edge_support.items():
        i = id_to_index[id_i]
        j = id_to_index[id_j]
        edge_mat[i, j] = cnt / B
        edge_mat[j, i] = cnt / B

    df_mean_sim = pd.DataFrame(mean_sim, index=out_labels, columns=out_labels)
    df_edge_sup = pd.DataFrame(edge_mat, index=out_labels, columns=out_labels)

    if not return_history:
        if return_label_map:
            return df_mean_sim, df_edge_sup, label_map
        return df_mean_sim, df_edge_sup

    history = {"mean_sim": hist_mean_sim, "edge_sup": hist_edge_sup}
    if track_bins:
        history["sampled_bins"] = hist_sampled_bins
        history["missing_bins"] = hist_missing_bins
    if return_label_map:
        history["label_map"] = label_map
        history["label_mode"] = lm

    return df_mean_sim, df_edge_sup, history
