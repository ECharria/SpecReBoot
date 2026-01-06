### - Imports
import sys, os

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from matchms.importing import load_from_mgf
from phylo2MS.phylo2ms.preprocessing.filtering import general_cleaning
from matchms import Spectrum

import numpy as np
import networkx as nx
import pandas as pd 

from phylo2MS.phylo2ms.binning.binning import global_bins, bin_spectra
from phylo2MS.phylo2ms.bootstrapping.bootstrapping import calculate_boostrapping
from phylo2MS.phylo2ms.networking.gnps_style import load_gnps_graph_and_id_map, add_rescued_edges_to_gnps_graph

from matchms.similarity import ModifiedCosine, CosineGreedy, FlashSimilarity

from phylo2MS.phylo2ms.networking.networking import build_base_graph, build_thresh_graph, build_core_rescue_graph

#load
spectra = list(load_from_mgf("/Users/rtlortega/Documents/PhD/Phylo2MS/phylo2MS/experiments/runs/SpecReboot_workflow_-_GNPS-39397e9f1b7547a79931dd9f57ec2e80-specs_ms.mgf"))


spectra_cleaned, report = general_cleaning(spectra,file_name = "pesticides_cleaned.mgf")

print(report)


decimals = 2
bins = global_bins(spectra_cleaned, decimals)        # don't overwrite function name
binned_spectra = bin_spectra(spectra_cleaned, decimals)

print("N spectra:", len(binned_spectra), "P bins:", len(bins))


df_mean_sim, df_edge_sup = calculate_boostrapping(
    binned_spectra,
    bins,
    B=100,  # try 1 first, then 10, then 50
    k=5,
    similarity_metric=ModifiedCosine(tolerance=0.02), # this part caon be substitute for example ModifiedCosine(tolerance=0.02)
    n_jobs=50,
)

df_mean_sim.to_csv("bootstrap_mean_similarity_Cos.csv")
df_edge_sup.to_csv("bootstrap_edge_support_Cos.csv")



# Load ans maps
gnps_network, id_map = load_gnps_graph_and_id_map(
    "/Users/rtlortega/Documents/PhD/Phylo2MS/phylo2MS/experiments/runs/SpecReboot_workflow_-_GNPS-39397e9f1b7547a79931dd9f57ec2e80-network_singletons.graphml",
    df_mean_sim.index,
    candidate_node_attrs="shared name",  # now OK because helper converts str -> (str,)
)

if not id_map:
    raise ValueError("Could not map bootstrap IDs to GNPS nodes. Check which node attribute matches df_mean_sim.index.")

G_rescued = add_rescued_edges_to_gnps_graph(
    gnps_network,
    df_mean_sim,
    df_edge_sup,
    id_map, 
    output_file="gnps_plus_rescued.graphml",
)

