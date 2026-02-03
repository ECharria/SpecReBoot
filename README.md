[![DOI](https://zenodo.org/badge/DOI/10.5281/https:/zenodo.org/records/18466798)](https://doi.org/10.5281/zenodo.18466798)

# SpecReBoot 🧬🔁  
*Statistical bootstrapping for spectral similarity and molecular networking*

> **Status:** in active development 🚧  
> Feedback, ideas, issues, and PRs are very welcome!

---

## What is SpecReBoot?

**SpecReBoot** brings the spirit of **phylogenetic bootstrapping** to **MS/MS molecular networking**.

In phylogenetics, bootstrapping asks: *“If I slightly perturb my data, do I recover the same relationships?”*  
SpecReBoot asks the same question for MS/MS spectra:

**“If I resample spectral features, do I recover the same edges in the network?”**

SpecReBoot generates pseudo-replicate spectra (via feature resampling), recomputes similarities across replicates, and reports **edge support** as a robustness measure for spectral relationships.

---

## What do you get?

For a dataset + similarity method, SpecReBoot produces:

- **Mean similarity matrix** (consensus similarity across replicates)
- **Edge support matrix** (how often an edge is recovered across replicates)
- **Networks (GraphML)**:
  - **Base network** (similarity threshold)
  - **Threshold network** (similarity + support + component-size constraints)
  - **Core–rescue network** (strict “core” edges + rescued edges)

This helps you:
- Filter unstable / fragile edges  
- Improve reproducibility across instruments and studies  
- Compare robustness across similarity methods

---

## Key ideas (in plain English)

- **Spectral features ≈ alignment positions** (but for fragments / losses / learned features)
- **Bootstrapping** = repeatedly resample features → pseudo-replicate spectra
- **Edge support** = fraction of replicates where two spectra are recovered as *mutual top-K neighbours*
- **Consensus network** = build using similarity **and** support thresholds

---

## Similarity scores supported

SpecReBoot can run multiple similarity methods so you can compare robustness across “classic” and learned scores:

- **Flash Cosine / Flash Modified Cosine**  
  Fast cosine-based scoring (fragment and hybrid matching). Great baseline and scalable.

- **Spec2Vec**  
  A machine-learning similarity that treats peaks like “words” and spectra like “documents”.  
  Uses a trained Word2Vec model to compare spectra by learned peak co-occurrence patterns.

- **MS2DeepScore (MS2DP)**  
  Deep learning embeddings for spectra. Similar spectra have nearby embeddings, allowing robust similarity even when peak overlap is imperfect.

> Note: Spec2Vec and MS2DeepScore require pre-trained models (paths passed via CLI).

---

## Installation (conda + editable install)

### 1) Install conda
If you don’t have conda yet, Miniconda is enough:  
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

### 2) Create the conda environment
From the **SpecReBoot repo root**:

```bash
conda env create -f environment.yml
conda activate specreboot
```

### 3) Install SpecReBoot 
First clone our repository where you want it:

```shell
git clone https://github.com/ECharria/SpecReBoot.git   
```
Then from this repo root type in the bash terminal: 

```shell
pip install -e .   
```
quick test:
```shell
specreboot --help
```

SpecReBoot is developed on Linux and macOS.

## Getting started 🚀

SpecReBoot provides a single command with two modes:
* matchms → full workflow: preprocessing → bootstrapping → network generation
* gnps → compute bootstrap + merge rescued edges into an existing GNPS GraphML network

Help is always available:

```shell
specreboot --help
specreboot matchms --help
specreboot gnps --help
```

### Mode 1 — matchms (full pipeline)
Runs:
1. preprocessing (general_cleaning)
2. binning
3. bootstrapping across multiple similarity scores
4. exports CSV + GraphML networks + runtime log

Example:
```shell
specreboot matchms \
  --mgf "/Users/rtlortega/.../RiPPs.mgf" \ #to configure
  --ms2dp-model "/Users/rtlortega/.../ms2deepscore_model.pt" \
  --spec2vec-model "/Users/rtlortega/.../Spec2Vec.model" \
  --outdir "/Users/rtlortega/.../out_matchms" \
  --prefix "RiPPs" \
  --B 30 --k 5 --n-jobs 4 \
  --sim-threshold 0.7 --sim-threshold-ms2dp 0.8
```

### Mode 2 — gnps (merge rescued edges into a GNPS network)
Use this mode when you already have a GNPS network (GraphML) and want to:
1. compute bootstrap support from your spectra
2. “rescue” supported edges
3. write GNPS + rescued as a new GraphML network

Example:
```shell
specreboot gnps \
  --mgf "/Users/rtlortega/.../specs_ms.mgf" \
  --gnps-graphml "/Users/rtlortega/.../network_singletons.graphml" \
  --outdir "/Users/rtlortega/.../out_gnps" \
  --prefix "pesticides" \
  --B 100 --k 5 --n-jobs 4 \
  --similarity modcos --tolerance 0.02 \
  --candidate-node-attrs "shared name" \
  --sim-core 0.7 --support-core 0.5 --sim-rescue-min 1e-5 --support-rescue 0.5
```

## Attribution
### License

The code in this package is licensed under the MIT License.

### Citation
Coming soon

### Contact
Please open a GitHub Issue for bugs / feature requests.
Maintainers: Rosina Torres-Ortega (rosina.torresortea@wur.nl) and Esteban Charria-Guiron (esteban.charriagiron@wur.nl)