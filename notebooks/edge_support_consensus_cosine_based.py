#!/usr/bin/env python
# coding: utf-8

# Phylo2MSClust

import matchms.filtering as msfilters
from matchms.importing import load_from_mgf

def spectrum_processing(s):
    s = msfilters.default_filters(s)
    s = msfilters.add_parent_mass(s)
    s = msfilters.normalize_intensities(s)
    s = msfilters.select_by_mz(s, mz_from=0, mz_to=1000)
    return s

# Load data from MGF file and apply filters
spectrums = [
    spectrum_processing(s)
    for s in load_from_mgf("/lustre/BIF/nobackup/charr003/projects/hypoxylaceae/MS2LDA/test_data/GNPS-NIH-NATURALPRODUCTSLIBRARY.mgf")
]
print(f"Number of spectra loaded: {len(spectrums)}")
print(spectrums[1])

# ---------------------------------------------------------------------
# Peak binning
# ---------------------------------------------------------------------
import numpy as np

decimals = 3  # change if needed

all_bins = []
for spec in spectrums:
    binned_mz = np.round(spec.peaks.mz, decimals)
    all_bins.extend(binned_mz)

global_bins = sorted(set(all_bins))

# ---------------------------------------------------------------------
# Rebuild spectra with binned peaks
# ---------------------------------------------------------------------
from matchms import Spectrum

list_of_binned_spectra = []
for spec in spectrums:
    mz_rounded = np.round(spec.peaks.mz, decimals)
    spectrum = Spectrum(
        mz=mz_rounded,
        intensities=spec.peaks.intensities,
        metadata=spec.metadata,
    )
    list_of_binned_spectra.append(spectrum)

# ---------------------------------------------------------------------
# Similarities
# ---------------------------------------------------------------------
from matchms import calculate_scores
from matchms.similarity import ModifiedCosine, CosineGreedy, FlashSimilarity

from collections import defaultdict, Counter
import pandas as pd
from joblib import parallel_backend

def calculate_boostrapping(
    list_of_binned_spectra,
    global_bins,
    B,
    k,
    spectra_similarity_metric,
    n_jobs=8,
    seed=42,
):
    """
    Bootstrapped spectral similarity with controllable parallelism.
    Returns mean similarity and edge support matrices as DataFrames.
    """
    rng = np.random.default_rng(seed)
    N = len(list_of_binned_spectra)
    P = len(global_bins)

    pair_sims_sum = defaultdict(float)
    pair_counts = defaultdict(int)
    edge_support = Counter()

    # be safe if some spectra don't have "scans"
    scan_ids = [
        s.get("scans")
        or s.metadata.get("scans")
        or s.metadata.get("spectrum_id")
        or s.metadata.get("title")
        or f"spec_{i}"
        for i, s in enumerate(list_of_binned_spectra)
    ]
    scan_to_idx = {sid: idx for idx, sid in enumerate(scan_ids)}

    for b in range(B):
        # ---- 1) sample bins ----
        sampled_bin_indices = rng.integers(0, P, size=P)
        sampled_bins = set(global_bins[i] for i in sampled_bin_indices)

        # ---- 2) build bootstrapped spectra ----
        spectra_boot = []
        for spec in list_of_binned_spectra:
            mz = spec.peaks.mz
            intens = spec.peaks.intensities
            mask = np.isin(mz, list(sampled_bins))
            kept_mz, kept_int = mz[mask], intens[mask]
            spectra_boot.append(
                Spectrum(
                    mz=kept_mz.astype("float32"),
                    intensities=kept_int.astype("float32"),
                    metadata=spec.metadata,
                )
            )

        if not spectra_boot:
            continue

        # ---- 3) compute similarities ----
        sim_matrix = np.zeros((N, N), dtype=float)
        with parallel_backend("loky", n_jobs=n_jobs):
            scores = calculate_scores(
                references=spectra_boot,
                queries=spectra_boot,
                similarity_function=spectra_similarity_metric,
                is_symmetric=True,
                # progress_bar=False,  # <-- removed for compatibility
            )

        for item in scores:
            # handle both old and new tuple formats
            if len(item) == 3:
                ref_spec, query_spec, score_value = item
            else:
                (ref_spec, query_spec), score_value = item

            i = scan_to_idx[ref_spec.get("scans")]
            j = scan_to_idx[query_spec.get("scans")]
            if i == j:
                continue
            sim_val = float(score_value[0])
            sim_matrix[i, j] = sim_matrix[j, i] = sim_val
            key = (scan_ids[i], scan_ids[j])
            pair_sims_sum[key] += sim_val
            pair_counts[key] += 1

        # ---- 4) mutual kNN edge support ----
        for i in range(N):
            idx_sorted = np.argsort(sim_matrix[i])[::-1]
            idx_sorted = idx_sorted[idx_sorted != i]
            knn_i = idx_sorted[:k]
            for j in knn_i:
                idx_sorted_j = np.argsort(sim_matrix[j])[::-1]
                idx_sorted_j = idx_sorted_j[idx_sorted_j != j]
                knn_j = idx_sorted_j[:k]
                if i in knn_j:
                    ekey = tuple(sorted((scan_ids[i], scan_ids[j])))
                    edge_support[ekey] += 1

        if (b + 1) % 5 == 0:
            print(f"[bootstrap {b+1}/{B}] done")

    # ---- aggregate ----
    mean_sim_mat = np.eye(N, dtype="float32")
    for (a, bkey), total_sim in pair_sims_sum.items():
        i, j = scan_to_idx[a], scan_to_idx[bkey]
        cnt = pair_counts[(a, bkey)]
        mean_sim_mat[i, j] = mean_sim_mat[j, i] = total_sim / cnt

    df_mean_sim = pd.DataFrame(mean_sim_mat, index=scan_ids, columns=scan_ids)

    edge_sup_mat = np.zeros((N, N), dtype="float32")
    for (a, bkey), cnt in edge_support.items():
        i, j = scan_to_idx[a], scan_to_idx[bkey]
        edge_sup_mat[i, j] = edge_sup_mat[j, i] = cnt / B

    df_edge_sup = pd.DataFrame(edge_sup_mat, index=scan_ids, columns=scan_ids)
    return df_mean_sim, df_edge_sup


# ---------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------
df_mean_sim, df_edge_sup = calculate_boostrapping(
    list_of_binned_spectra,
    global_bins,
    B=100,  # try 1 first, then 10, then 50
    k=5,
    spectra_similarity_metric=ModifiedCosine(tolerance=0.02),
    n_jobs=50,
)

df_mean_sim.to_csv("bootstrap_mean_similarity_ModCos.csv")
df_edge_sup.to_csv("bootstrap_edge_support_ModCos.csv")
print("Done.")