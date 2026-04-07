import numpy as np
from matchms import Spectrum
from typing import Any
from tqdm import tqdm
import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import time

def calculate_bootstrapping(
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
    """Calculate bootstrapped mean similarity and mutual-kNN edge support for a single replicate.

    Parameters
    -------
    spectra_binned (list[Spectrum]): List of binned spectra.
    global_bins (array-like): Global m/z bins used as the bootstrap sampling space.
    B : int
        Number of bootstrap replicates.
    k : int
        Number of nearest neighbors used for mutual-kNN edge support.
    similarity_metric : callable
        Similarity object used to calculate pairwise similarities.
    n_jobs : int
        Number of workers used for parallel batch execution.
    batch_size : int
        Number of bootstrap replicates processed per batch.
    seed : int
        Base random seed for bootstrap resampling.
    return_history : bool
        Whether to return cumulative bootstrap history.
    track_bins : bool
        Whether to store sampled and missing bins for each bootstrap.
    label_mode : str
        Output label type. Options are "feature", "scan", or "internal".
    return_label_map : bool
        Whether to return the mapping between output labels and
        original spectrum identifiers.
    verbose : bool
        Whether to print progress and timing information.

    Returns
    -------
    tuple:
    - If return_history is False and return_label_map is False:
      (df_mean_sim, df_edge_sup)
    - If return_history is False and return_label_map is True:
      (df_mean_sim, df_edge_sup, label_map)
    - If return_history is True:
      (df_mean_sim, df_edge_sup, history)
    """
    total_start = time.perf_counter()

    # NOTE: due to floating point arithmatic, the resulting values in the cosine similarity matrix might differ with the orignal function with around <1e-7. 
    history = {}

    # The similarity implementation expects a NumPy array of global bins.
    global_bins = np.array(global_bins)

    # Build output labels and label map for the selected label mode.
    out_labels, label_info = _get_spectra_labels(label_mode, spectra_binned)

    dataset_size = len(spectra_binned)

    # Accumulate results across all bootstrap replicates.
    total_pair_similarities = np.zeros((dataset_size, dataset_size), dtype=float)  # matrix with the sum of all similarities of each pair combination
    total_pair_counts       = np.zeros((dataset_size, dataset_size), dtype=float)  # matrix with the nr of times a given pair is used in the bootstrapping
    total_edge_support      = np.zeros((dataset_size, dataset_size), dtype=float)  # matrix with the nr of times a given pair are each others closest k neighbours

    bootstrap_ids = list(range(B))
    batches = [bootstrap_ids[i:i + batch_size] for i in range(0, B, batch_size)]

    if verbose:
        print(f"Running {B} bootstraps in {len(batches)} batches (batch_size={batch_size}) with {n_jobs} workers", flush=True)

    compute_start = time.perf_counter()

    args = [(spectra_binned, global_bins, similarity_metric, seed, k, b, return_history, track_bins, verbose) for b in batches]

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(lambda x: bootstrap_batch(*x), args))

    compute_end = time.perf_counter()

    if verbose:
        print(f"Bootstrap batch execution finished in {compute_end - compute_start:.2f} seconds", flush=True)

    all_history = []

    merge_start = time.perf_counter()

    for result in results:
        pair_similarities, pair_counts, edge_support, batch_history = result

        # Add the batch results to the global totals.
        total_pair_similarities += pair_similarities
        total_pair_counts       += pair_counts
        total_edge_support      += edge_support
        all_history             += batch_history

    merge_end = time.perf_counter()
    total_end = time.perf_counter()

    if verbose:
        print(f"Merge finished in {merge_end - merge_start:.2f} seconds", flush=True)
        print(f"Total bootstrapping completed in {total_end - total_start:.2f} seconds", flush=True)


     
    # Calculate the mean similarity only where a pair was observed.
    mean_similarities = np.divide(
        total_pair_similarities,
        total_pair_counts,
        out=np.zeros_like(total_pair_similarities, dtype=float),
        where=total_pair_counts != 0
    )
    # Self-similarity is fixed to 1 to match the original implementation.
    np.fill_diagonal(mean_similarities , 1)
    mean_edge_support = total_edge_support / B

    # Needed to match correct return type
    df_mean_sim = pd.DataFrame(mean_similarities, index=out_labels, columns=out_labels)
    df_edge_sup = pd.DataFrame(mean_edge_support, index=out_labels, columns=out_labels)

    if return_history:
        if verbose:
            print("Reconstructing cumulative history from per-bootstrap results...", flush=True)

        replay_start = time.perf_counter()

        history_mean_sim, history_edge_sup = _reconstruct_history(dataset_size, all_history)
        history["mean_sim"] = history_mean_sim
        history["edge_sup"] = history_edge_sup

        replay_end = time.perf_counter()

        if verbose:
            print(f"History reconstruction finished in {replay_end - replay_start:.2f} seconds", flush=True)

    total_end = time.perf_counter()

    print(f"Total bootstrapping completed in {total_end - total_start:.2f} seconds", flush=True)

    if track_bins:
        history["sampled_bins"] = [h["sampled_bins"] for h in sorted(all_history, key=lambda x: x["b"])]
        history["missing_bins"] = [h["missing_bins"] for h in sorted(all_history, key=lambda x: x["b"])]

    if return_history:
        if return_label_map:
            history |= label_info
        history = {k: v for k, v in history.items() if len(v) != 0}
        return df_mean_sim, df_edge_sup, history

    if return_label_map:
        return df_mean_sim, df_edge_sup, label_info["label_map"]
    return df_mean_sim, df_edge_sup



def bootstrap_batch(                
    spectra_binned,
    global_bins,
    similarity_metric,
    seed,
    k,
    B,
    collect_history=False,
    track_bins=False,
    verbose=False,
):
    """Run a batch of bootstrap replicates.

    Parameters
    -------
    spectra_binned : list[Spectrum]
        List of binned spectra.
    global_bins : np.ndarray
        Global m/z bins used for resampling.
    similarity_metric: callable
        Similarity object used to calculate pairwise similarities.
    seed : int
        Base random seed.
    k : int
        Number of nearest neighbors used for mutual-kNN support.
    B : list[int]
        Bootstrap replicate indices processed in this batch.
    collect_history : bool
        Whether to store per-bootstrap results.
    track_bins : bool
        Whether to store sampled and missing bins for each bootstrap.
    verbose : bool
        Whether to print timing information.    

    Returns
    -------
    tuple:
    (total_pair_similarities, total_pair_counts, total_edge_support, history)
    """
    t_batch_start = time.perf_counter()

    dataset_size = len(spectra_binned)
    random_generator = np.random.default_rng(seed)

    # Accumulate results across the bootstrap replicates in this batch.
    total_pair_similarities = np.zeros((dataset_size, dataset_size), dtype=float)  # matrix with the sum of all similarities of each pair combination
    total_pair_counts       = np.zeros((dataset_size, dataset_size), dtype=float)  # matrix with the nr of times a given pair is used in the bootstrapping
    total_edge_support      = np.zeros((dataset_size, dataset_size), dtype=float)  # matrix with the nr of times a given pair are each others closest k neighbours

    history = []

    for b in tqdm(B):

        masked_spectra, sampled_bins = _mask_spectra(random_generator, global_bins, spectra_binned)

        # This is different now, but matches the internal workings of calculate_scores from matchms module
        pair_similarity_matrix = similarity_metric.matrix(masked_spectra, masked_spectra, array_type="numpy", is_symmetric=True)

        top_k_nearest_neighbours = mutual_topk(pair_similarity_matrix, k)

        # Get the position of the pairs that we used in this iteration
        pair_counts  = (pair_similarity_matrix != 0).astype(int)
        edge_support = (top_k_nearest_neighbours != 0).astype(int)

        # Add the results if this iteration
        total_pair_similarities += pair_similarity_matrix
        total_pair_counts       += pair_counts
        total_edge_support      += edge_support

        run = {}
        history.append(run)
        if collect_history:
            
            run["b"] = b
            run["pair_sim_sum"] = pair_similarity_matrix
            run["pair_counts"]  = pair_counts
            run["edge_support"] = edge_support


        if track_bins:
            used_bins   = np.unique(sampled_bins)
            unused_bins = np.setdiff1d(global_bins, sampled_bins)

            run["sampled_bins"] = used_bins
            run["missing_bins"] = unused_bins

    t_batch_end = time.perf_counter()
    if verbose:
        print(f"Batch {B[0]}-{B[-1]} finished in {t_batch_end - t_batch_start:.2f} s", flush=True)

    return (
        total_pair_similarities, 
        total_pair_counts,        
        total_edge_support,    
        history
    )
     

def _reconstruct_history(dataset_size, all_history):
    """Reconstruct cumulative similarity and edge-support history.

    Parameters:
    dataset_size : int
        Number of spectra in the dataset.
    all_history :list[dict]
        Per-bootstrap history entries.

    Returns:
    tuple[list[np.ndarray], list[np.ndarray]]:
    Cumulative mean similarity matrices and cumulative edge-support matrices.
    """
    history_mean_sim = []
    history_edge_sup = []

    current_pair_similarities = np.zeros((dataset_size, dataset_size), dtype=float)  # matrix with the sum of all similarities of each pair combination
    current_pair_counts       = np.zeros((dataset_size, dataset_size), dtype=float)  # matrix with the nr of times a given pair is used in the bootstrapping
    current_edge_support      = np.zeros((dataset_size, dataset_size), dtype=float)  # matrix with the nr of times a given pair are each others closest k neighbours

    for b, history in enumerate(sorted(all_history, key=lambda x: x["b"])):
        current_pair_similarities += history["pair_sim_sum"]
        current_pair_counts       += history["pair_counts"] 
        current_edge_support      += history["edge_support"]

        current_mean_similarities = np.divide(
            current_pair_similarities,
            current_pair_counts,
            out=np.zeros_like(current_pair_similarities, dtype=float),
            where=current_pair_counts != 0
        )
        current_mean_edge_support = current_edge_support / (b + 1)

        history_mean_sim.append(current_mean_similarities)
        history_edge_sup.append(current_mean_edge_support)

    return history_mean_sim, history_edge_sup



def _get_spectra_labels(label_mode: str, spectra_binned: list[Spectrum]) -> tuple[list[str], dict[str, any]]:  # Logic from previous main function refactored into a helper function
    """Create output labels and label map for the selected label mode.

    Parameters:
    label_mode : str
        Label mode. Options are "feature", "scan", or "internal".
    spectra_binned : list[Spectrum]
        List of binned spectra.

    Returns:
    tuple[list[str], dict[str, any]]:
        Output labels and a dictionary containing label_map and label_mode.
    """
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
    scan_labels_u = __make_unique(scan_labels)
    feature_labels_u = __make_unique(feature_labels)
    internal_ids_u = __make_unique(internal_ids)

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
    label_info = dict(label_map=label_map, label_mode=lm)
    return out_labels, label_info


def __make_unique(labels: list[any]) -> list[str]:
    """Make labels unique by appending a suffix to duplicated values."""
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


def mutual_topk(A: np.ndarray, k: int) -> np.ndarray:
    """Return a binary matrix marking mutual top-k neighbours.

    Parameters:
    A (np.ndarray): Pairwise similarity matrix.
    k (int): Number of nearest neighbours to retain per row.

    Returns:
    np.ndarray: Binary matrix where 1 indicates a mutual top-k relationship.
    """
    n = A.shape[0]
    A_work = A.copy()
    np.fill_diagonal(A_work, -np.inf)  # This makes sure self-similarity is not counted as a top neighbor

    row_sorted = np.argsort(A_work, axis=1)  # Gives use the indices if the sorted values: [3, 2, 7, 1] -> [3, 1, 0, 2] (along one axis)
    row_sorted = row_sorted[:, ::-1]  # Reverse the indices, so the index of the highest nr is first
    row_topk = row_sorted[:, :k]  # Remove the part after the k-best neighbour

    row_mask = np.zeros_like(A_work, dtype=bool)
    rows = np.arange(n)[:, None]  # Create a range of nrs [0, 1, 2, .., n] and make it 2d -> [[0, 1, 2, .., n]], needed to make mask
    row_mask[rows, row_topk] = True  # Sets only the top neighbors to True

    mutual_mask = row_mask & row_mask.T  # Only a value that is in *mutual* top k neighbors kept

    result = np.zeros_like(A)  
    result[mutual_mask] = 1  # sets all the mutual top k neighbors to 1
    return result


def _mask_spectra(random_generator: Any, global_bins: np.ndarray, binned_spectra: list[Spectrum]) -> tuple[list[Spectrum], np.ndarray]:
    """Mask spectra according to one bootstrap sample of the global bins.

    Parameters:
    random_generator : Any
        NumPy random generator used for resampling.
    global_bins : np.ndarray
        Global m/z bins used for bootstrap sampling.
    binned_spectra : list[Spectrum]
        List of binned spectra.

    Returns:
    tuple[list[Spectrum], np.ndarray]:
    Masked spectra and the sampled bins used in this bootstrap replicate.
    """
    result = []

    sampled_indices = random_generator.integers(0, len(global_bins), size=len(global_bins))

    sampled_bins = global_bins[sampled_indices]
    sampled_bins = np.unique(sampled_bins)

    for index, spectrum in enumerate(binned_spectra):
        mask = np.isin(spectrum.peaks.mz, sampled_bins)

        mz = spectrum.peaks.mz[mask] 
        intensities = spectrum.peaks.intensities[mask]

        mz = mz.astype("float32")
        intensities = intensities.astype("float32")

        if len(mz) == 0:
            mz = np.array([ global_bins[0] ], dtype="float32")
            intensities = np.array([0.0],     dtype="float32")

        masked_spectrum = Spectrum(mz, intensities, spectrum.metadata)
        result.append(masked_spectrum)

    return result, sampled_bins
    