# PEdictor-code
Deep cfDNA fragmentomics analysis for early prediction of preeclampsia
# PEdictor: Deep cfDNA Fragmentomics Analysis for Early Prediction of Preeclampsia

This repository contains the complete analysis pipeline used in the study **“PEdictor: Deep cfDNA Fragmentomics Decoding Enables Efficient Prediction of Early Preeclampsia Risk”**, including fragment length profiling, CNV analysis, end-motif computation, K-mer sequence analysis, and machine-learning–based prediction models.

All scripts are modular and can be run independently.  
**All paths must be defined by the user.**  
Each script performs one specific analysis step.

---

## 📁 **Repository Structure**

| File | Description |
|------|-------------|
| `01_1_fragment_length.py` | Compute cfDNA fragment length distribution (100–500 bp), short/long fragment ratios, and autosome-wide fragment counts. |
| `01_2_quantity.py` | Quantify cfDNA fragment counts per sample or per genomic bin. |
| `02_1_chrs_CNV.py` | Chromosome-level CNV profiling using variable-bin windows (50 kb / 80 kb / 150 kb / 200 kb). |
| `02_2_genes_CNV.py` | Gene-level CNV profiling and significance filtering. |
| `03_end_motif.py` | Compute frequencies of all 256 possible 4-bp end motifs. |
| `04_kmer.py` | Perform k-mer (3–9 mer) profiling for short (100–150 bp) and long (150–200 bp) fragments. |
| `05_LASSO.py` | Feature selection using LASSO regression for all feature types. |
| `06_TURF_IFS.py` | Feature ranking via TURF and incremental feature selection. |
| `07_stacked_classifier.py` | Train ML/ensemble classifiers (SVM, KNN, RF, XGB, MLP) and evaluate performance metrics. |

---

## 🧬 **Analysis Overview**

This pipeline reproduces all analyses described in the manuscript:

### **1. Fragment Length Analysis**
- cfDNA fragments (100–500 bp) quantified from WGS BAM files.
- PE samples show fewer short fragments (100–150 bp) and enrichment of long fragments (150–200 bp).
- LASSO selects core fragmentation features and trains multiple classifiers.

### **2. CNV Analysis**
- Chromosome-arm CNV profiling with different window sizes.
- Gene-level CNV computing and pathway enrichment.
- LASSO identifies chromosome-level and gene-level discriminative CNVs.

### **3. End-Motif Analysis**
- Computes frequencies of all 4-bp terminal motifs.
- LASSO selects key motifs with the strongest PE-control differences.

### **4. K-mer Sequence Profiling**
- K-mer profiling of 3–9 bp motifs.
- Short-fragment 8-mers achieve highest classification performance.
- PWMs generated for MEME/TOMTOM motif matching.

### **5. Machine Learning & Ensemble Modeling**
- Models: SVM, KNN, RF, XGB, MLP.
- TURF for feature ranking, SHAP for interpretability.
- 7 high-confidence 8-mer markers enable robust prediction.

---

## ⚙️ **Environment Requirements**

You may use conda or pip to install dependencies.

