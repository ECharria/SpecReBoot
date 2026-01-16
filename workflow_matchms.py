### - Imports
import sys, os
import pickle

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from matchms.importing import load_from_mgf
from phylo2MS.phylo2ms.preprocessing.filtering import general_cleaning
from matchms import Spectrum

import numpy as np

from phylo2MS.phylo2ms.binning.binning import global_bins, bin_spectra
from phylo2MS.phylo2ms.bootstrapping.bootstrapping import calculate_boostrapping
from matchms.similarity import ModifiedCosine, CosineGreedy, FlashSimilarity

from phylo2MS.phylo2ms.networking.networking import build_base_graph, build_thresh_graph, build_core_rescue_graph

import networkx as nx



### - Load and preprocess
spectra = load_from_mgf("/Users/rtlortega/Documents/PhD/Datasets/Manually collected RiPPs NPATLAS_GNPS.mgf")

spectra_cleaned, report = general_cleaning(spectra,file_name = "Manually collected RiPPs NPATLAS_GNPS.mgf_cleaned.mgf")

print(report)



### - Peak binning
decimals = 2  # change if needed
global_bins = global_bins(spectra_cleaned, decimals)

### - Rebuild spectra with binned peaks
list_of_binned_spectra = bin_spectra(spectra_cleaned, decimals)


## define similarity
from ms2deepscore.models import load_model
from ms2deepscore import MS2DeepScore
ms2deepscore_model = load_model("/Users/rtlortega/Documents/PhD/WP2/Code/LATTICE_Project/models/ms2deepscore_model.pt") #change to your path where you have your model
ms2deepscore_similarity = MS2DeepScore(ms2deepscore_model)


df_mean_sim, df_edge_sup = calculate_boostrapping(
    list_of_binned_spectra,
    global_bins,
    B=100,  # try 1 first, then 10, then 50
    k=5,
    similarity_metric=ms2deepscore_similarity, # this part caon be substitute for example ModifiedCosine(tolerance=0.02)
    n_jobs=50,
    track_bins=True
)

df_mean_sim.to_csv("bootstrap_mean_similarity_Cos.csv")
df_edge_sup.to_csv("bootstrap_edge_support_Cos.csv")

    
# Networking
build_base_graph(df_mean_sim, df_edge_sup) #base similarity
build_thresh_graph(df_mean_sim, df_edge_sup, sim_threshold = 0.7, support_threshold = 0.5) #two thresholds

build_core_rescue_graph(df_mean_sim, df_edge_sup,sim_core = 0.7, support_core = 0.3, sim_rescue_min = 0.00001 ,support_rescue = 0.5, output_file="ms2dp_core_rescued_RIPPs.graphml")

