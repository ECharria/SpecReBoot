import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from matchms import calculate_scores
from matchms import Spectrum
from concurrent.futures import ThreadPoolExecutor
import time


def _run_single_bootstrap(
    b,
    spectra_binned,
    global_bins_arr,
    internal_ids,
    id_to_index,
    similarity_metric,
    seed,
    k,
):
    rng = np.random.default_rng(seed + b)

    N = len(spectra_binned)
    P = len(global_bins_arr)

    pair_sim_sum = defaultdict(float)
    pair_counts = defaultdict(int)
    edge_support = Counter()

    sampled_indices = rng.integers(0, P, size=P)
    sampled_bins_arr = np.unique(global_bins_arr[sampled_indices])

    spectra_boot = []

    for idx, (internal_id, spec) in enumerate(zip(internal_ids, spectra_binned)):
        mz = spec.peaks.mz
        intens = spec.peaks.intensities

        mask = np.isin(mz, sampled_bins_arr)
        mz_kept = mz[mask]
        int_kept = intens[mask]

        if mz_kept.size == 0:
            dummy_mz = float(global_bins_arr.max() + 1000.0 + idx)
            mz_kept = np.array([dummy_mz], dtype="float32")
            int_kept = np.array([1.0], dtype="float32")

        meta = {**spec.metadata, "internal_id": internal_id}

        spectra_boot.append(
            Spectrum(
                mz=mz_kept.astype("float32"),
                intensities=int_kept.astype("float32"),
                metadata=meta,
            )
        )

    sim_matrix = np.zeros((N, N), dtype=float)

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

    # mutual kNN
    k_eff = min(k, N - 1)
    all_knn = []

    for i in range(N):
        row = sim_matrix[i].copy()
        row[i] = -np.inf
        knn_i = np.argpartition(row, -k_eff)[-k_eff:]
        knn_i = knn_i[np.argsort(row[knn_i])[::-1]]
        all_knn.append(set(knn_i))

    for i in range(N):
        for j in all_knn[i]:
            if i < j and i in all_knn[j]:
                key = tuple(sorted((internal_ids[i], internal_ids[j])))
                edge_support[key] += 1

    return pair_sim_sum, pair_counts, edge_support


def run_bootstrap_batch(
    batch_bootstrap_ids,
    spectra_binned,
    global_bins_arr,
    internal_ids,
    id_to_index,
    similarity_metric,
    seed,
    k,
    B,
    verbose=False,
):
    batch_pair_sim_sum = defaultdict(float)
    batch_pair_counts = defaultdict(int)
    batch_edge_support = Counter()

    t_batch_start = time.perf_counter()

    for b in batch_bootstrap_ids:
        ps, pc, es = _run_single_bootstrap(
            b=b,
            spectra_binned=spectra_binned,
            global_bins_arr=global_bins_arr,
            internal_ids=internal_ids,
            id_to_index=id_to_index,
            similarity_metric=similarity_metric,
            seed=seed,
            k=k,
        )

        for key, val in ps.items():
            batch_pair_sim_sum[key] += val

        for key, val in pc.items():
            batch_pair_counts[key] += val

        for key, val in es.items():
            batch_edge_support[key] += val

        if verbose and ((b + 1) % 10 == 0 or (b + 1) == B):
            print(f"[bootstrap {b+1}] done", flush=True)

    t_batch_end = time.perf_counter()
    if verbose:
        print(
            f"Batch {batch_bootstrap_ids[0]+1}-{batch_bootstrap_ids[-1]+1} "
            f"finished in {t_batch_end - t_batch_start:.2f} s",
            flush=True,
        )

    return batch_pair_sim_sum, batch_pair_counts, batch_edge_support


def calculate_boostrapping(
    spectra_binned,
    global_bins,
    B=100,
    k=5,
    similarity_metric=None,
    n_jobs=4,
    batch_size=10,
    seed=42,
    return_history=False,
    track_bins=False,
    label_mode="feature",
    return_label_map=True,
    verbose: bool = True,
):
    if similarity_metric is None:
        raise ValueError("similarity_metric must be provided")

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    batch_size

    if return_history:
        raise NotImplementedError(
            "return_history=True is not supported with batched parallel bootstrapping."
        )

    if global_bins is None or not hasattr(global_bins, "__len__"):
        raise TypeError(
            "global_bins must be an array/list of bin m/z values."
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

    N = len(spectra_binned)
    global_bins_arr = np.asarray(global_bins)

    pair_sim_sum = defaultdict(float)
    pair_counts = defaultdict(int)
    edge_support = Counter()

    scan_labels = []
    feature_labels = []
    internal_ids = []
    id_to_index = {}

    for idx, spec in enumerate(spectra_binned):
        md = spec.metadata

        scan_number = (
            getattr(spec, "get", lambda x: None)("scans")
            or md.get("scans")
            or md.get("SCAN_NUMBER")
            or f"scan_{idx}"
        )

        scan_labels.append(str(scan_number))

        feature_id = md.get("feature_id", f"feat_{idx}")
        feature_labels.append(str(feature_id))

        internal_id = f"INTFID_{feature_id}_{idx}"
        internal_ids.append(internal_id)
        id_to_index[internal_id] = idx

    scan_labels_u = make_unique(scan_labels)
    feature_labels_u = make_unique(feature_labels)
    internal_ids_u = make_unique(internal_ids)

    lm = label_mode.lower()

    if lm == "scan":
        out_labels = scan_labels_u
    elif lm == "internal":
        out_labels = internal_ids_u
    else:
        out_labels = feature_labels_u

    label_map = pd.DataFrame(
        {
            "out_label": out_labels,
            "scan": scan_labels,
            "scan_unique": scan_labels_u,
            "feature_id": feature_labels,
            "feature_unique": feature_labels_u,
            "internal_id": internal_ids,
        }
    )

    # Build bootstrap batches
    bootstrap_ids = list(range(B))
    batches = [
        bootstrap_ids[i:i + batch_size]
        for i in range(0, B, batch_size)
    ]

    args = [
        (
            batch,
            spectra_binned,
            global_bins_arr,
            internal_ids,
            id_to_index,
            similarity_metric,
            seed,
            k,
            B,
            verbose,
        )
        for batch in batches
    ]

    total_start = time.perf_counter()
    if verbose:
        print(
            f"Running {B} bootstraps in {len(batches)} batches "
            f"(batch_size={batch_size}) with {n_jobs} workers",
            flush=True,
        )

    compute_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(lambda x: run_bootstrap_batch(*x), args))
    compute_end = time.perf_counter()

    if verbose:
        print(
            f"Bootstrap batch execution finished in {compute_end - compute_start:.2f} seconds",
            flush=True,
        )
        
    
    if verbose:
        print("Merging batch results...", flush=True)
    
    merge_start = time.perf_counter()

    for ps, pc, es in results:
        for key, val in ps.items():
            pair_sim_sum[key] += val

        for key, val in pc.items():
            pair_counts[key] += val

        for key, val in es.items():
            edge_support[key] += val

    merge_end = time.perf_counter()
    total_end = time.perf_counter()

    if verbose:
        print(f"Merge finished in {merge_end - merge_start:.2f} seconds", flush=True)
        print(f"Total bootstrapping completed in {total_end - total_start:.2f} seconds", flush=True)

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

    if return_label_map:
        return df_mean_sim, df_edge_sup, label_map

    return df_mean_sim, df_edge_sup