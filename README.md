# Phylo2MS  
*Statistical bootstrapping for spectral similarity and molecular networking*

> **Status:** Early development 🚧
> Feedback, ideas, and contributions are very welcome!

---

## Overview

**Phylo2MS** brings the statistical rigor of **phylogenetic bootstrapping** to **mass spectrometry (MS/MS)**.  
Inspired by *Felsenstein’s algorithm* in phylogenetics, phylo2MS resamples fragment peaks, neutral losses, or learned spectral features to generate *pseudo-replicate spectra*.  

By recalculating similarity scores (currently cosine-based) across many replicates, phylo2MS estimates **bootstrap support** for every edge between spectra, quantifying how robust molecular relationships are across feature resampling.  

The resulting **support matrices** and **consensus networks** help:

- Filter unreliable connections between spectra  
- Improve reproducibility across instruments and studies  
- Identify statistically irrelevant fragments (linking to the *dark metabolome* discussion)

---

## 🧠 Key ideas

- **Fragment peaks ≈ alignment sites** → resampled like columns in sequence bootstrapping  
- **Edge support:** fraction of bootstraps where two spectra are mutual top-K neighbors  
- **Consensus network:** built from edges above a support threshold  
- **Feature-level lens:** measures fragment importance

---
