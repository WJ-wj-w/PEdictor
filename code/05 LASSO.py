#!/usr/bin/env python3
"""
LASSO Feature Selection and Data Preprocessing Pipeline
--------------------------------------------------------
Description:
    This script performs standardization, LASSO-based feature selection,
    and train/test splitting for a high-dimensional dataset.
    It outputs selected features and optionally visualizes LASSO coefficients.

Requirements:
    - Python 3
    - pandas
    - numpy
    - scikit-learn
    - matplotlib (optional, for plotting)

Usage:
    1. Set your input CSV file path, feature names, and output folder.
    2. Run the script: python lasso_feature_selection_pipeline.py
    3. Outputs:
        - Console: list of selected features
        - CSV: selected features for training data
        - Optional: LASSO coefficient plot
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

# ---------------- User Configuration ----------------
INPUT_CSV = "/path/to/input_data.csv"    # CSV file with features and target variable
FEATURE_NAMES = None                     # Optional list of feature names (including target as last element)
TARGET_COLUMN = None                     # Optional, if using DataFrame column name for target
OUTPUT_FOLDER = "/path/to/output_folder"
TEST_SIZE = 0.3
RANDOM_STATE = 0
PLOT_COEFFICIENTS = True                 # Set to False to skip plotting

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------- Load Data ----------------
df = pd.read_csv(INPUT_CSV)

# If feature names are provided, assign them
if FEATURE_NAMES is not None:
    df.columns = FEATURE_NAMES

# ---------------- Split features and target ----------------
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # Last column as target variable

# ---------------- Standardize features ----------------
colNames = X.columns
X = X.astype(np.float64)
X_scaled = StandardScaler().fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=colNames)

# ---------------- Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ---------------- LASSO Feature Selection ----------------
lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=1000000)
lasso.fit(X_train, y_train)

# Select features with non-zero coefficients
selected_model = SelectFromModel(lasso, prefit=True)
X_train_selected = selected_model.transform(X_train)
selected_feature_names = X_train.columns[selected_model.get_support()]

print("Selected features:", list(selected_feature_names))

# ---------------- Save Selected Features ----------------
selected_df = pd.DataFrame(X_train_selected, columns=selected_feature_names)
selected_df[TARGET_COLUMN if TARGET_COLUMN else 'Target'] = y_train.reset_index(drop=True)
output_file = os.path.join(OUTPUT_FOLDER, "X_train_selected_features.csv")
selected_df.to_csv(output_file, index=False)
print(f"Selected features saved to: {output_file}")