# -*- coding: utf-8 -*-
"""
cfDNA fragment length analysis
Processes BED files to compute fragment counts by chromosome and fragment size,
merges sample data for machine learning, and prepares data for plotting.
"""

import os
import pandas as pd
import glob
import re
from concurrent.futures import ThreadPoolExecutor
import subprocess

# ----------------------------
# Step 1: Process BED files
# ----------------------------
def process_file(file_path, output_dir):
    """
    Filter fragments by length (100-150bp and 150-200bp) and count per chromosome.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(file_path) + '_length_filtered_chr_count.csv')

    # Bedtools commands for fragment length filtering
    bedtools_cmd_100_150 = (
        f"awk '{{if ($3 - $2 >= 100 && $3 - $2 <= 150) print $0}}' {file_path} | "
        "bedtools groupby -g 1 -c 2 -o count"
    )
    bedtools_cmd_150_200 = (
        f"awk '{{if ($3 - $2 > 150 && $3 - $2 <= 200) print $0}}' {file_path} | "
        "bedtools groupby -g 1 -c 2 -o count"
    )

    # Execute commands
    result_100_150 = subprocess.run(bedtools_cmd_100_150, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    result_150_200 = subprocess.run(bedtools_cmd_150_200, shell=True, stdout=subprocess.PIPE, universal_newlines=True)

    # Convert to DataFrame
    df_100_150 = pd.read_csv(pd.compat.StringIO(result_100_150.stdout), sep='\t', header=None, names=['chr', 'count_100_150'])
    df_150_200 = pd.read_csv(pd.compat.StringIO(result_150_200.stdout), sep='\t', header=None, names=['chr', 'count_150_200'])
    df_merged = pd.merge(df_100_150, df_150_200, on='chr', how='outer').fillna(0)
    df_merged.to_csv(output_file, index=False)

def process_directory(directory, output_dir, threads=10):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.bed')]
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(process_file, f, output_dir) for f in files]
        for future in futures:
            future.result()

# ----------------------------
# Step 2: Summarize counts by main chromosome
# ----------------------------
def summarize_chromosome_counts(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))

    def get_main_chromosome(chrom):
        return chrom.split('_')[0]

    for file in csv_files:
        df = pd.read_csv(file)
        df['main_chr'] = df['chr'].apply(get_main_chromosome)
        summary_df = df.groupby('main_chr')[['count_100_150', 'count_150_200']].sum().reset_index()
        df_summary = df[df['chr'] == df['main_chr']].merge(summary_df, on='main_chr', suffixes=('', '_summed'))
        df_summary['count_100_150'] = df_summary['count_100_150_summed']
        df_summary['count_150_200'] = df_summary['count_150_200_summed']
        df_summary = df_summary.drop(columns=['count_100_150_summed', 'count_150_200_summed'])
        output_file = os.path.join(output_dir, os.path.basename(file).replace(".csv", "_summary.csv"))
        df_summary.to_csv(output_file, index=False)

# ----------------------------
# Step 3: Split fragment sizes
# ----------------------------
def split_fragment_size(input_dir, output_dir, size='100_150'):
    os.makedirs(output_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(input_dir, '*_summary.csv'))

    for file in csv_files:
        df = pd.read_csv(file)
        if size == '100_150':
            df = df.drop(columns=['count_150_200', 'main_chr'], errors='ignore')
        elif size == '150_200':
            df = df.drop(columns=['count_100_150', 'main_chr'], errors='ignore')
        output_file = os.path.join(output_dir, os.path.basename(file))
        df.to_csv(output_file, index=False)

# ----------------------------
# Step 4: Merge all samples for machine learning
# ----------------------------
def merge_samples(input_dir, output_file):
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    combined_data = pd.DataFrame()
    number_pattern = re.compile(r'(\d+P\d+)')
    for file in csv_files:
        df = pd.read_csv(file)
        match = number_pattern.search(os.path.basename(file))
        df['type'] = match.group(1) if match else "unknown"
        combined_data = pd.concat([combined_data, df], ignore_index=True)
    combined_data.to_csv(output_file, index=False)

# ----------------------------
# Step 5: Plotting (simplified)
# ----------------------------
import matplotlib.pyplot as plt

def plot_cfDNA_stack_bar(data_dict, chromosomes, output_file):
    """
    data_dict: {'PE_100_150': df1, 'CTRL_150_200': df2, ...}
    chromosomes: list of chromosomes to plot
    """
    mean_values = {k: v[chromosomes].mean() for k, v in data_dict.items()}
    # Create stacked bar plot
    x = range(len(chromosomes))
    bottom = pd.Series([0]*len(chromosomes))
    for key, values in mean_values.items():
        plt.bar(x, values.values, bottom=bottom, label=key)
        bottom += values
    plt.xticks(x, chromosomes, rotation=45)
    plt.ylabel('Average cfDNA Quantity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
