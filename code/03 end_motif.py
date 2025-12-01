#!/usr/bin/env python3
"""
cfRNA 4-mer End Sequence Analysis Pipeline
------------------------------------------
Description:
    This script processes BAM files to extract the last 4 bases (4-mer) of each read,
    computes 4-mer frequency matrices, and saves the results as CSV files. It is 
    designed for cfRNA/cfDNA fragmentomic analysis.

Requirements:
    - Python 3
    - pysam
    - pandas
    - numpy
    - psutil

Usage:
    1. Set your own input BAM folder and output folder paths below.
    2. Run the script: python cfRNA_4mer_pipeline.py
    3. Outputs:
        - Text files containing read-end sequences
        - CSV files with 4-mer frequencies and relative frequencies
"""

import os
import pysam
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

# ---------------- System Configuration ----------------
def get_system_info():
    """
    Returns the number of threads to use.
    Users can modify this manually if needed.
    """
    cpu_count = 10
    return cpu_count

# ---------------- Core Functions ----------------
def extract_end_sequences(bam_file, output_file):
    """
    Extract the last 4 bases of each read in a BAM file and write to a text file.
    """
    with pysam.AlignmentFile(bam_file, 'rb') as bam:
        with open(output_file, 'w') as output:
            for read in bam:
                if not read.is_reverse:
                    output.write(read.query_sequence[-4:] + '\n')  # last 4 bases

def generate_4mer_frequency_dataframe(file_path):
    """
    Generate a dataframe containing counts and relative frequencies of 4-mers.
    """
    print(f"Generating 4-mer frequency dataframe for file: {file_path}")
    with open(file_path, 'r') as file:
        end_sequences = file.read().splitlines()

    kmer_freq = Counter(end_sequences)
    total_count = sum(kmer_freq.values())

    # Compute index for each 4-mer (optional for downstream analysis)
    kmer_index = []
    for sequence in kmer_freq.keys():
        index = sum([4 ** i * "ACGT".index(j) for i, j in enumerate(sequence)])
        kmer_index.append(index)

    kmer_data = {'Motif': list(kmer_freq.keys()), 
                 'Count': list(kmer_freq.values()), 
                 'Index': kmer_index}
    df = pd.DataFrame(kmer_data)
    df['Relative_Freq'] = df['Count'] / total_count

    return df

def process_bam_file(bam_file, output_folder, sample_type):
    """
    Process a single BAM file: extract end sequences and compute 4-mer frequencies.
    """
    output_path = os.path.join(output_folder, f"{sample_type}_{os.path.basename(bam_file).replace('.bam', '_end_sequences.txt')}")
    print(f"Extracting end sequences for file: {bam_file}")
    extract_end_sequences(bam_file, output_path)
    print(f"Generating 4-mer frequency dataframe for file: {output_path}")
    kmer_frequency_df = generate_4mer_frequency_dataframe(output_path)
    return os.path.basename(bam_file).replace('.bam', ''), kmer_frequency_df

def process_bam_folder(bam_folder, output_folder, sample_type):
    """
    Process all BAM files in a folder in parallel and return a list of (sample_name, dataframe) tuples.
    """
    os.makedirs(output_folder, exist_ok=True)
    result_matrix = []

    bam_files = [os.path.join(bam_folder, f) for f in os.listdir(bam_folder) if f.endswith(".bam")]
    max_workers = get_system_info()

    print(f"Processing {len(bam_files)} BAM files with {max_workers} threads.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_bam_file, bam_file, output_folder, sample_type): bam_file for bam_file in bam_files}
        for future in futures:
            sample_name, df = future.result()
            result_matrix.append((sample_name, df))

    return result_matrix

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    # User-defined input and output paths
    CTRL_bam_folder = '/path/to/CTRL_bam_folder'  # folder containing control BAM files
    output_folder = '/path/to/output_folder'      # folder to save results

    # Process control samples
    CTRL_matrices = process_bam_folder(CTRL_bam_folder, output_folder, sample_type="CTRL")

    # Save 4-mer frequency matrices as CSV
    for sample_name, df in CTRL_matrices:
        file_path = os.path.join(output_folder, f"CTRL_{sample_name}_kmer.csv")
        print(f"Writing CSV file for sample {sample_name}: {file_path}")
        df.to_csv(file_path, index=False, columns=['Motif', 'Count', 'Relative_Freq'])

    print("4-mer end sequence analysis completed successfully!")
