#!/usr/bin/env python3
"""
cfDNA Feature Selection and SHAP Analysis Pipeline
--------------------------------------------------
Description:
    This script performs the following tasks for cfDNA high-dimensional data:
    1. Load and standardize data
    2. Perform TURF scoring and Iterative Feature Selection (IFS)
    3. Save selected features and accuracy history


Requirements:
    - Python 3
    - pandas
    - numpy
    - scikit-learn
    - matplotlib
    - shap

Usage:
    1. Set all input/output paths in the User Configuration section.
    2. Run the script: python cfRNA_feature_selection_pipeline.py
    3. Outputs:
        - CSV of selected features and accuracy history
        - SHAP plots per model
        - Accuracy vs number of features plot
"""

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ===========================
# User Configuration
# ===========================
INPUT_DATA = "/path/to/211.csv"
FEATURE_SELECTION_RESULTS = "/path/to/7_results.csv"
OUTPUT_FOLDER = "/path/to/output_folder"
TEST_SIZE = 0.3
RANDOM_STATE = 42
MAX_ITER_FEATURES = 211
PLOT_SHAP = True

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===========================
# 1. Load and preprocess data
# ===========================
df = pd.read_csv(INPUT_DATA)

# Move target column "type" to the last position
cols = list(df.columns)
cols.remove('type')
cols.append('type')
df = df[cols]

# Split features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Standardize features
X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ===========================
# 2. TURF scoring function
# ===========================
def compute_coverage(X_train, y_train):
    """Compute coverage scores (correlation with target)"""
    coverage = [np.corrcoef(X_train[feature], y_train)[0, 1] for feature in X_train.columns]
    return np.array(coverage)

def turf_scoring(X_train, y_train):
    """Rank features by TURF score"""
    coverage_scores = compute_coverage(X_train, y_train)
    sorted_indices = np.argsort(coverage_scores)[::-1]
    sorted_features = X_train.columns[sorted_indices]
    return sorted_features, coverage_scores[sorted_indices]

# ===========================
# 3. Iterative Feature Selection (IFS)
# ===========================
def iterative_feature_selection_turf(X_train, y_train, model, cv, sorted_features, max_iter=MAX_ITER_FEATURES):
    selected_features = []
    best_score = 0
    scores_history = []

    for i in range(min(len(sorted_features), max_iter)):
        scores = []
        for feature in sorted_features:
            temp_features = selected_features + [feature]
            X_temp = X_train[temp_features]

            cv_scores = []
            for train_idx, val_idx in cv.split(X_temp, y_train):
                X_fold_train, X_fold_val = X_temp.iloc[train_idx], X_temp.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                model.fit(X_fold_train, y_fold_train)
                y_pred = model.predict(X_fold_val)
                cv_scores.append(accuracy_score(y_fold_val, y_pred))

            scores.append((np.mean(cv_scores), feature))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        if scores[0][0] > best_score:
            best_score = scores[0][0]
            selected_features.append(scores[0][1])
        scores_history.append(best_score)
    
    return selected_features, scores_history

# ===========================
# 4. Model definition and selection
# ===========================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores_history_dict = {}
selected_features_dict = {}

for model_name, model in models.items():
    print(f"\nRunning TURF+IFS for {model_name}...")
    sorted_features, _ = turf_scoring(X_train, y_train)
    selected_features, scores_history = iterative_feature_selection_turf(X_train, y_train, model, cv, sorted_features)
    scores_history_dict[model_name] = scores_history
    selected_features_dict[model_name] = selected_features
    print(f"Best score: {scores_history[-1]}, Features: {selected_features}")

# ===========================
# 5. Save results to CSV
# ===========================
results = []
for model_name in models.keys():
    for i, score in enumerate(scores_history_dict[model_name]):
        features_subset = selected_features_dict[model_name][:i+1]
        results.append({
            "Model": model_name,
            "Number_of_Features": i+1,
            "Accuracy": score,
            "Selected_Features": ", ".join(features_subset)
        })

results_df = pd.DataFrame(results)
results_csv_path = os.path.join(OUTPUT_FOLDER, "7_results.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"Results saved to {results_csv_path}")