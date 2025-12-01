#!/usr/bin/env python3
"""
cfRNA Fragment k-mer Analysis Pipeline (k=3~9, short and long fragments)
------------------------------------------------------------------------
Description:
    This script processes BAM files to extract fragment sequences for two fragment size ranges:
        - Short: 100-150 bp
        - Long: 150-200 bp
    It computes k-mer frequencies (k=3~9), generates summary matrices for each fragment range,
    and performs Min-Max normalization. Designed for cfRNA/cfDNA fragmentomics.

Requirements:
    - Python 3
    - pysam
    - pandas
    - scikit-learn

Usage:
    1. Set your input BAM folder, output folder, fragment length ranges, and k values.
    2. Run the script: python cfRNA_kmer_pipeline_short_long.py
    3. Outputs:
        - Filtered BAM files for each fragment length range
        - CSV files with k-mer counts for each sample and fragment length
        - Summary k-mer matrices across samples for each fragment length
        - Normalized k-mer matrices
"""

import os
import pysam
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------- User Configuration ----------------
BAM_FOLDER = "/path/to/input_bam_folder"          # folder containing BAM files
OUTPUT_FOLDER = "/path/to/output_folder"          # folder to save results
FRAGMENT_RANGES = {
    "short": (100, 150),
    "long": (150, 200)
}
KMER_RANGE = range(3, 10)                         # k values for k-mer (3~9)
NUM_THREADS = 10                                  # threads for filtering BAMs
FILES_PER_PROCESS = 1                              # number of files per process in multiprocessing

# ---------------- Utility Functions ----------------
def filter_fragments(bam_file, output_folder, min_len, max_len):
    """Filter BAM file fragments by length and save filtered BAM with index."""
    output_file = os.path.join(output_folder, os.path.basename(bam_file).replace('.bam', f'_{min_len}-{max_len}bp_filtered.bam'))
    os.makedirs(output_folder, exist_ok=True)
    with pysam.AlignmentFile(bam_file, "rb") as in_bam, \
         pysam.AlignmentFile(output_file, "wb", header=in_bam.header) as out_bam:
        for read in in_bam.fetch():
            if min_len <= read.query_length <= max_len:
                out_bam.write(read)
    pysam.index(output_file)
    return output_file

def extract_kmers(sequence, k):
    """Extract all k-mers of length k from a sequence."""
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)] if sequence else []

def count_kmers_in_bam(bam_file, k):
    """Count k-mers for a single BAM file."""
    kmer_counts = Counter()
    if not os.path.exists(bam_file + ".bai"):
        pysam.index(bam_file)
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        for read in bam.fetch():
            kmer_counts.update(extract_kmers(read.query_sequence, k))
    return kmer_counts

def save_kmer_counts(kmer_counts, output_file):
    """Save k-mer counts to CSV."""
    df = pd.DataFrame(list(kmer_counts.items()), columns=['k-mer', 'count'])
    df.to_csv(output_file, index=False)

def process_kmer_for_file(bam_file, k, output_dir):
    """Process a BAM file for one k and save the counts."""
    counts = count_kmers_in_bam(bam_file, k)
    output_file = os.path.join(output_dir, os.path.basename(bam_file).replace('.bam', f'_k{k}_counts.csv'))
    save_kmer_counts(counts, output_file)

def process_files_parallel(bam_files, k, output_dir, num_processes=10, files_per_process=1):
    """Process multiple BAM files in parallel for a given k."""
    os.makedirs(output_dir, exist_ok=True)
    chunks = [bam_files[i:i + files_per_process] for i in range(0, len(bam_files), files_per_process)]
    with Pool(num_processes) as pool:
        pool.starmap(lambda chunk: [process_kmer_for_file(f, k, output_dir) for f in chunk], [(chunk,) for chunk in chunks])

def summarize_kmer_csv(input_folder, output_file):
    """Merge individual k-mer CSVs into one summary matrix."""
    csv_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('_counts.csv')]
    summary_df = pd.DataFrame()
    for f in csv_files:
        df = pd.read_csv(f)
        sample_name = os.path.basename(f).split('_k')[0]
        df = df.rename(columns={'count': sample_name}).set_index('k-mer')
        summary_df = df if summary_df.empty else summary_df.join(df, how='outer')
    summary_df = summary_df.reset_index()
    summary_df.to_csv(output_file, index=False)

def normalize_matrix(input_csv, output_csv):
    """Perform Min-Max normalization on numeric columns."""
    df = pd.read_csv(input_csv)
    numeric_cols = df.columns[1:]
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    df.to_csv(output_csv, index=False)

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    bam_files = [os.path.join(BAM_FOLDER, f) for f in os.listdir(BAM_FOLDER) if f.endswith('.bam')]

    for frag_type, (min_len, max_len) in FRAGMENT_RANGES.items():
        print(f"Processing {frag_type} fragments ({min_len}-{max_len} bp)...")
        filtered_bam_folder = os.path.join(OUTPUT_FOLDER, f"{frag_type}_filtered_bam")
        os.makedirs(filtered_bam_folder, exist_ok=True)

        # Step 1: Filter BAMs by fragment length
        filtered_bams = []
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = [executor.submit(filter_fragments, f, filtered_bam_folder, min_len, max_len) for f in bam_files]
            for future in futures:
                filtered_bams.append(future.result())

        # Step 2: k-mer counting for k=3~9
        for k in KMER_RANGE:
            print(f"Counting {k}-mers for {frag_type} fragments...")
            kmer_output_dir = os.path.join(OUTPUT_FOLDER, f"{frag_type}_k{k}_counts")
            process_files_parallel(filtered_bams, k, kmer_output_dir, num_processes=NUM_THREADS)

            # Step 3: Summarize individual CSVs into a single matrix
            summary_file = os.path.join(OUTPUT_FOLDER, f"{frag_type}_summary_k{k}.csv")
            summarize_kmer_csv(kmer_output_dir, summary_file)

            # Step 4: Normalize summary matrix
            normalized_file = os.path.join(OUTPUT_FOLDER, f"{frag_type}_summary_k{k}_normalized.csv")
            normalize_matrix(summary_file, normalized_file)

    print("k-mer analysis pipeline for short and long fragments completed successfully!")
