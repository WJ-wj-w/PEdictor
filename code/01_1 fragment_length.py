"""
Analysis of cfDNA fragment length hallmarks in preeclampsia (PE) and controls.
Generates:
- Fragment size density curves
- Proportions of 100–150 bp, 150–200 bp fragments, and short/long ratio

Inputs:
- Folder containing BED files for PE samples
- Folder containing BED files for Control samples
- Output directory (figures + temp pickle files)

All input/output locations must be provided by the user before running.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cfDNApipe import *

# ------------------------------
# User-defined paths
# ------------------------------
pe_bed_folder = "PATH_TO_PE_BED_FILES"
ctrl_bed_folder = "PATH_TO_CONTROL_BED_FILES"
output_folder = "OUTPUT_DIRECTORY"

os.makedirs(output_folder, exist_ok=True)

# ------------------------------
# Pipeline configuration
# ------------------------------
pipeConfigure(
    threads=25,
    genome="hg38",
    refdir="PATH_TO_REFERENCE_GENOME",
    outdir=output_folder,
    data="WGS",
    type="paired",
    JavaMem="10G",
    build=True,
)

verbose = False

pe_bed_files = glob.glob(os.path.join(pe_bed_folder, "*.bed"))
ctrl_bed_files = glob.glob(os.path.join(ctrl_bed_folder, "*.bed"))

# ------------------------------
# Compute fragment length densities
# ------------------------------
res_fraglenplot_comp = fraglenplot_comp(
    casebedInput=pe_bed_files,
    ctrlbedInput=ctrl_bed_files,
    labelInput=["PE", "Control"],
    caseupstream=True,
    verbose=verbose,
)

# ------------------------------
# Helper: find peak fragment position
# ------------------------------
def get_peak_position(keys, vals):
    peak_idx = np.argmax(vals)
    return keys[peak_idx]

# ------------------------------
# Load pickle files for PE and Control
# ------------------------------
pe_pickle_dir = os.path.join(output_folder, "PE")
ctrl_pickle_dir = os.path.join(output_folder, "Control")

def compute_peak_positions(folder):
    peak_positions = []
    for fname in os.listdir(folder):
        if not fname.endswith(".pickle"):
            continue
        with open(os.path.join(folder, fname), "rb") as f:
            d = dict(sorted(pickle.load(f).items()))
        keys = np.array(list(d.keys()))
        vals = np.array(list(d.values()))
        vals = vals / vals.sum()
        peak_positions.append(get_peak_position(keys, vals))
    return peak_positions

pe_peak_positions = compute_peak_positions(pe_pickle_dir)
ctrl_peak_positions = compute_peak_positions(ctrl_pickle_dir)

# ------------------------------
# Compute fragment proportions and short/long ratio
# ------------------------------
short_range = (100, 150)
long_range = (150, 200)

def compute_proportions(folder):
    short_list, long_list = [], []
    for fname in os.listdir(folder):
        if not fname.endswith(".pickle"):
            continue
        with open(os.path.join(folder, fname), "rb") as f:
            d = dict(sorted(pickle.load(f).items()))
        keys = np.array(list(d.keys()))
        vals = np.array(list(d.values()))
        vals = vals / vals.sum()
        short_list.append(vals[(keys >= short_range[0]) & (keys <= short_range[1])].sum())
        long_list.append(vals[(keys >= long_range[0]) & (keys <= long_range[1])].sum())
    return short_list, long_list

pe_short, pe_long = compute_proportions(pe_pickle_dir)
ctrl_short, ctrl_long = compute_proportions(ctrl_pickle_dir)

ratio_pe = [s/l if l > 0 else 0 for s, l in zip(pe_short, pe_long)]
ratio_ctrl = [s/l if l > 0 else 0 for s, l in zip(ctrl_short, ctrl_long)]

# ------------------------------
# Create DataFrames
# ------------------------------
short_df = pd.DataFrame({
    "Category": ["PE"] * len(pe_short) + ["Control"] * len(ctrl_short),
    "Proportion": pe_short + ctrl_short
})

long_df = pd.DataFrame({
    "Category": ["PE"] * len(pe_long) + ["Control"] * len(ctrl_long),
    "Proportion": pe_long + ctrl_long
})

ratio_df = pd.DataFrame({
    "Category": ["PE"] * len(ratio_pe) + ["Control"] * len(ratio_ctrl),
    "Ratio": ratio_pe + ratio_ctrl
})

# ------------------------------
# Basic plotting
# ------------------------------
fig_dir = os.path.join(output_folder, "fragment_plots")
os.makedirs(fig_dir, exist_ok=True)

# Density curves (Figure 2a)
plt.figure()
sns.lineplot(data=short_df, x="Category", y="Proportion")
plt.savefig(os.path.join(fig_dir, "fragment_density.pdf"))
plt.close()

# Short, Long, Ratio boxplots (Figure 2b)
plt.figure()
sns.boxplot(data=short_df, x="Category", y="Proportion")
plt.savefig(os.path.join(fig_dir, "short_fragments.pdf"))
plt.close()

plt.figure()
sns.boxplot(data=long_df, x="Category", y="Proportion")
plt.savefig(os.path.join(fig_dir, "long_fragments.pdf"))
plt.close()

plt.figure()
sns.boxplot(data=ratio_df, x="Category", y="Ratio")
plt.savefig(os.path.join(fig_dir, "short_long_ratio.pdf"))
plt.close()
