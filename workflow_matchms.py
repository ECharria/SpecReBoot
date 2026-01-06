### - Imports
import sys, os

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from matchms.importing import load_from_mgf
from phylo2MS.phylo2ms.preprocessing.filtering import general_cleaning
from matchms import Spectrum

import numpy as np

from phylo2MS.phylo2ms.binning.binning import global_bins, bin_spectra
from phylo2MS.phylo2ms.bootstrapping.bootstrapping import calculate_boostrapping
from.phylo2ms.networking.gnps_style import load_gnps_graph_and_id_map, add_rescued_edges_to_gnps_graph
from matchms.similarity import ModifiedCosine, CosineGreedy, FlashSimilarity

from phylo2MS.phylo2ms.networking.networking import build_base_graph, build_thresh_graph, build_core_rescue_graph



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
)

df_mean_sim.to_csv("bootstrap_mean_similarity_Cos.csv")
df_edge_sup.to_csv("bootstrap_edge_support_Cos.csv")

import networkx as nx
# Load ans maps
gnps_network, id_map = load_gnps_graph_and_id_map("/Users/rtlortega/Documents/PhD/Phylo2MS/phylo2MS/experiments/runs/SpecReboot_workflow_-_GNPS-39397e9f1b7547a79931dd9f57ec2e80-network_singletons.graphml",
                           df_mean_sim.index,
                           candidate_node_attrs="shared name")

#Do the third recued network
add_rescued_edges_to_gnps_graph(gnps_network, df_mean_sim, df_edge_sup)