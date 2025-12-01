"""
Analysis of cfDNA copy-number variations (CNV) in preeclampsia (PE) and control samples.
Generates:
- Genome-wide CNV Z-scores
- Combined Z-score tables across multiple bin sizes (50 kb, 80 kb, 150 kb, 200 kb, 500 kb)
- Sample type annotation (PE vs Control)

Inputs:
- Folder containing BAM files for PE samples
- Folder containing BAM files for Control samples
- Reference genome directory
- Output directory

All input/output locations must be provided by the user before running.
"""

import os
import glob
import pandas as pd
from cfDNApipe import *

# ------------------------------
# User-defined paths
# ------------------------------
pe_bam_folder = "PATH_TO_PE_BAM_FILES"
ctrl_bam_folder = "PATH_TO_CONTROL_BAM_FILES"
ref_dir = "PATH_TO_REFERENCE_GENOME"
output_dir = "OUTPUT_DIRECTORY"

os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# Pipeline configuration
# ------------------------------
pipeConfigure2(
    threads=40,
    genome="hg38",
    refdir=ref_dir,
    outdir=output_dir,
    data="WGS",
    type="paired",
    JavaMem="32G",
    case="PE",
    ctrl="CTRL",
    build=True,
)

verbose = False

# ------------------------------
# Load BAM files
# ------------------------------
case_bam = glob.glob(os.path.join(pe_bam_folder, "*.bam"))
ctrl_bam = glob.glob(os.path.join(ctrl_bam_folder, "*.bam"))

# ------------------------------
# Case sample processing
# ------------------------------
switchConfigure("PE")
case_bamCounter = bamCounter(
    bamInput=case_bam,
    chromsizeInput=os.path.join(ref_dir, "hg38.chrom.sizes"),
    upstream=True,
    verbose=verbose,
    threads=40,
    binlen=50000,
    stepNum="case01"
)

case_gcCounter = runCounter(
    fileInput=[os.path.join(ref_dir, "hg38.fa")],
    filetype=0,  # GC content counter
    upstream=True,
    verbose=verbose,
    threads=40,
    binlen=50000,
    stepNum="case02"
)

case_GCCorrect = GCCorrect(
    readupstream=case_bamCounter,
    gcupstream=case_gcCounter,
    readtype=2,
    corrkey="/",
    verbose=verbose,
    threads=40,
    stepNum="case03"
)

# ------------------------------
# Control sample processing
# ------------------------------
switchConfigure("CTRL")
ctrl_bamCounter = bamCounter(
    bamInput=ctrl_bam,
    chromsizeInput=os.path.join(ref_dir, "hg38.chrom.sizes"),
    upstream=True,
    verbose=verbose,
    threads=40,
    binlen=50000,
    stepNum="ctrl01"
)

ctrl_gcCounter = runCounter(
    fileInput=[os.path.join(ref_dir, "hg38.fa")],
    filetype=0,
    upstream=True,
    verbose=verbose,
    threads=40,
    binlen=50000,
    stepNum="ctrl02"
)

ctrl_GCCorrect = GCCorrect(
    readupstream=ctrl_bamCounter,
    gcupstream=ctrl_gcCounter,
    readtype=2,
    corrkey="/",
    verbose=verbose,
    threads=40,
    stepNum="ctrl03"
)

# ------------------------------
# Compute CNV
# ------------------------------
res_computeCNV = computeCNV(
    caseupstream=case_GCCorrect,
    ctrlupstream=ctrl_GCCorrect,
    stepNum="ARMCNV",
    threads=40,
    verbose=verbose
)

# ------------------------------
# Process Z-score files across multiple bin sizes
# ------------------------------
bin_sizes = ["50kb", "80kb", "150kb", "200kb", "500kb"]
all_dfs = []

for bin_size in bin_sizes:
    zscore_file = os.path.join(output_dir, f"{bin_size}_Z-score.txt")
    df = pd.read_csv(zscore_file, sep="\t")
    df.to_csv(os.path.join(output_dir, f"{bin_size}_Z-score.csv"), index=False)
    
    # Transpose Z-score table
    df_transposed = df.transpose()
    df_transposed.to_csv(os.path.join(output_dir, f"{bin_size}_Z-score_transposed.csv"), header=False)
    
    # Add bin size column
    df_transposed['bins'] = bin_size
    all_dfs.append(df_transposed)

# ------------------------------
# Combine all bin sizes
# ------------------------------
combined_df = pd.concat(all_dfs, ignore_index=True)

# ------------------------------
# Annotate sample types
# ------------------------------
def assign_type(sample_name, case_files, ctrl_files):
    if sample_name in ctrl_files:
        return 'CTRL'
    elif sample_name in case_files:
        return 'PE'
    else:
        return None

sample_names = combined_df.iloc[:, 0]
case_files = [os.path.basename(f).replace(".bam","") for f in case_bam]
ctrl_files = [os.path.basename(f).replace(".bam","") for f in ctrl_bam]

combined_df['type'] = sample_names.apply(lambda x: assign_type(x, case_files, ctrl_files))

# ------------------------------
# Save final combined Z-score table
# ------------------------------
combined_df.to_csv(os.path.join(output_dir, "combined_Z-score_bins_type.csv"), index=False)
