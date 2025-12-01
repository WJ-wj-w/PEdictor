
"""
stacked_classifier_pipeline.py

Functionality:
---------------
This script performs feature standardization, train-test split, 
hyperparameter tuning for multiple classifiers (SVM, KNN, Random Forest, XGBoost, MLP), 
and combines them in a stacking classifier using Logistic Regression as the meta-classifier. 
It computes accuracy and AUC for both training and test sets.

Input:
------
- CSV file with features and target column named 'type'.
  The last column is considered as the target variable.

Output:
-------
- Prints best hyperparameters for each base model
- Prints accuracy and AUC for training and test sets
- Optional: plotting ROC curves (if needed)
- Trained stacking classifier ready for predictions

Usage:
------
python stacked_classifier_pipeline.py --input /path/to/7-mer.csv

"""

import pandas as pd
import numpy as np
import argparse

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from sklearn.base import BaseEstimator, ClassifierMixin

# ----------------------------
# Command line arguments
# ----------------------------
parser = argparse.ArgumentParser(description='Stacked classifier for binary classification')
parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
args = parser.parse_args()

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(args.input)

# Move 'type' column to the last position if needed
if 'type' in df.columns:
    columns = [c for c in df.columns if c != 'type'] + ['type']
    df = df[columns]

# Split features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Feature names
feature_names = X.columns.tolist()

# Standardize features
X = pd.DataFrame(StandardScaler().fit_transform(X), columns=feature_names)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# ----------------------------
# Base models with tuned parameters
# ----------------------------
best_svc = SVC(random_state=0, C=0.5, kernel='rbf', probability=True)
best_knn = KNeighborsClassifier(n_neighbors=16, weights='uniform')
best_rfc = RandomForestClassifier(random_state=0, max_depth=5, min_samples_leaf=2, min_samples_split=2, n_estimators=4)
best_xgb = XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='error', learning_rate=1, max_depth=1, n_estimators=6)

# ----------------------------
# MLP Keras wrapper
# ----------------------------
def create_mlp_model(optimizer='Adam', units=16, dropout_rate=0.1, l2_reg=0.01):
    model = Sequential([
        Dense(units, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(units, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(units, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn, epochs=32, batch_size=16, **kwargs):
        self.build_fn = build_fn
        self.kwargs = kwargs
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y):
        self.model = self.build_fn(**self.kwargs)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)

    def predict_proba(self, X):
        probs = self.model.predict(X)
        return np.concatenate([1 - probs, probs], axis=1)

best_mlp = KerasClassifierWrapper(build_fn=create_mlp_model, optimizer='Adam', units=16, dropout_rate=0.1, l2_reg=0.01, epochs=32, batch_size=16)

# ----------------------------
# Stacking classifier
# ----------------------------
base_classifiers = [
    ('svm', best_svc),
    ('knn', best_knn),
    ('rf', best_rfc),
    ('xgb', best_xgb),
    ('mlp', best_mlp)
]

meta_classifier = LogisticRegression()
stacking_classifier = StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier)
stacking_classifier.fit(X_train, y_train)

# ----------------------------
# Evaluate performance
# ----------------------------
train_probs = stacking_classifier.predict_proba(X_train)[:, 1]
test_probs = stacking_classifier.predict_proba(X_test)[:, 1]

train_pred = stacking_classifier.predict(X_train)
test_pred = stacking_classifier.predict(X_test)

print("Training AUC:", roc_auc_score(y_train, train_probs))
print("Training Accuracy:", accuracy_score(y_train, train_pred))
print("Test AUC:", roc_auc_score(y_test, test_probs))
print("Test Accuracy:", accuracy_score(y_test, test_pred))
