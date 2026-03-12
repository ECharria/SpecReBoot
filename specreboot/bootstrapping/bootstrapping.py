import numpy as np
from matchms import Spectrum
from joblib import parallel_backend
from typing import Any
from tqdm import tqdm
import pandas as pd
from collections import Counter

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
    global_bins = np.array(global_bins)  # Important, a simple list will raise an error

    out_labels, label_info = _get_spectra_labels(label_mode, spectra_binned)
    history = dict(mean_sim=[], edge_sup=[], sampled_bins=[], missing_bins=[]) | label_info  # Just saves all metadata

    dataset_size = len(spectra_binned)
    random_generator = np.random.default_rng(seed)

    total_pair_similarities = np.zeros((dataset_size, dataset_size), dtype=float)  # matrix with the sum of all similarities of each pair combination
    total_pair_counts       = np.zeros((dataset_size, dataset_size), dtype=float)  # matrix with the nr of times a given pair is used in the bootstrapping
    total_edge_support      = np.zeros((dataset_size, dataset_size), dtype=float)  # matrix with the nr of times a given pair are each others closest k neighbours

    for b in tqdm(range(B)):

        masked_spectra, sampled_bins = _mask_spectra(random_generator, global_bins, spectra_binned)

        # This is different now, but matches the internal workings of calculate_scores from matchms module
        with parallel_backend("loky", n_jobs=n_jobs):
            similarity_matrix = similarity_metric.matrix(masked_spectra, masked_spectra, array_type="numpy", is_symmetric=True)

        top_k_nearest_neighbours = mutual_topk(similarity_matrix, k)

        # Get the position of the pairs that we used in this iteration
        similarity_matrix_binary        = (similarity_matrix != 0).astype(int)
        # ^ This line requires attention:
        #   currently, we're skipping all 0 values, because the orignial function iterates over a SparseMatrix that ALSO skips values that are 0. 
        #   However, it's possible that a pair of spectra that both *not* empty could still result in a similarity score of 0. Currently, that is not counted, but maybe it should be. 
        #   In other words, if spec A and spec B had a similarity that was 0.01, they are counted, but if they happen to have a similarity of 0.00, they are not counted. 
        #   Is this intended behaviour?


        top_k_nearest_neighbours_binary = (top_k_nearest_neighbours != 0).astype(int)

        # Add the results if this iteration
        total_pair_similarities += similarity_matrix
        total_pair_counts       += similarity_matrix_binary
        total_edge_support      += top_k_nearest_neighbours_binary

        if return_history:
            # Get the average of similarities
            current_similarities = np.divide(
                total_pair_similarities, 
                total_pair_counts, 
                out=np.zeros_like(total_pair_similarities, 
                dtype=float),
                where=total_pair_counts != 0
            )
            current_edge_support = total_edge_support / (b + 1)

            history["mean_sim"].append(current_similarities)
            history["edge_sup"].append(current_edge_support)

        if track_bins:
            used_bins   = np.unique(sampled_bins)
            unused_bins = np.setdiff1d(global_bins, sampled_bins)

            history["sampled_bins"].append(used_bins)
            history["missing_bins"].append(unused_bins)
     

    # Get the average of similarities
    mean_similarities = np.divide(
        total_pair_similarities,
        total_pair_counts,
        out=np.zeros_like(total_pair_similarities, dtype=float),
        where=total_pair_counts != 0
    )
    np.fill_diagonal(mean_similarities , 1)  # needed to exactly match original implementation; all spectra have a similarity of 1 to themselves
    mean_edge_support = total_edge_support / B

    # Needed to match correct return type
    df_mean_sim = pd.DataFrame(mean_similarities, index=out_labels, columns=out_labels)
    df_edge_sup = pd.DataFrame(mean_edge_support, index=out_labels, columns=out_labels)

    # Return history according to settings
    if return_label_map and not return_history:
        return df_mean_sim, df_edge_sup, history["label_map"]
    
    if not return_history:
        return df_mean_sim, df_edge_sup
    
    history = {k: v for k, v in history.items() if len(v) != 0}   # purge all missing data
    return df_mean_sim, df_edge_sup, history



def _get_spectra_labels(label_mode: str, spectra_binned: list[Spectrum]) -> tuple[list[str], dict[str, any]]:  # Logic from previous main function refactored into a helper function
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
    """ gives a matrix in the shap of A, where a 1 means a pair are each others top-k neighbor """

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
    