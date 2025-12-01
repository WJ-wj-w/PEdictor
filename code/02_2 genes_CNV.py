#!/usr/bin/env python3
"""
cfDNApipe CNV Analysis Pipeline - GitHub-ready Version
--------------------------------
Description:
    Simplified, ready-to-run CNV analysis script using cfDNApipe.

Requirements:
    - Python 3
    - cfDNApipe installed
    - BAM files for cases and controls
    - Reference genome files (hg38)

Usage:
    1. Modify input/output paths and BAM file lists.
    2. Run: python cfDNApipe_cnv_pipeline.py
    3. Outputs: CNV batch results, plots, and tables.
"""

from cfDNApipe import *

# ---------------- Pipeline Configuration ----------------
pipeConfigure(
    threads=30,
    genome="hg38",
    refdir=r"/path/to/reference",
    outdir=r"/path/to/output",
    data="WGS",
    type="paired",
    build=True,
    JavaMem="10G",
)

verbose = False  # True for detailed logging

# ---------------- Input BAM files ----------------
case_bams = ["/path/to/case1.bam", "/path/to/case2.bam"]
ctrl_bams = ["/path/to/control1.bam", "/path/to/control2.bam"]

# ---------------- CNV Analysis ----------------
res_cnvbatch = cnvbatch(
    casebamInput=case_bams,
    ctrlbamInput=ctrl_bams,
    caseupstream=True,
    access="/path/to/access-mappable.hg38.bed",
    annotate="/path/to/refFlat_hg38.txt",
    verbose=verbose,
    stepNum="CNV01",
)

res_cnvPlot = cnvPlot(upstream=res_cnvbatch, verbose=verbose, stepNum="CNV02")
res_cnvTable = cnvTable(upstream=res_cnvbatch, verbose=verbose, stepNum="CNV03")

print("CNV analysis completed successfully!")
