"""
Final XGBoost model training and evaluation using LOSO cross-validation.

This script trains and evaluates the final XGBoost model with predetermined
hyperparameters across all subjects using Leave-One-Subject-Out cross-validation.
"""

import json
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold

from scripts.weartime_ml_training_evaluation.training.utils import load_json_to_dataframe_flat

# Loading the metadata. This includes the reference and the data length for each ID and sensor
with open("ref_metadata.json") as file:
    ref_metadata = json.load(file)
# Unique data IDs
subject_ids = list(set([value["id"] for value in ref_metadata.values()]))

# Load data
data = load_json_to_dataframe_flat("features_xgboost.json")

# Separate features and target
features = data.drop(columns=["id", "sensor", "segment_id", "label", "window_center_idx"], errors="ignore")
target = data["label"]
groups = data["id"]

# Hardcoded best parameters (final params)
best_params = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "random_state": 42,
}

# LOSO cross-validation
gkf = GroupKFold(n_splits=data["id"].nunique())

# Storage for results
fold_results = []
all_y_test = []
all_y_pred = []

print("=== LOSO Cross-Validation ===\n")

# Start timing
total_start = time.time()

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(features, target, groups)):
    X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

    test_subject = groups.iloc[test_idx].unique()[0]

    # Train model with fixed parameters
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Store results
    fold_results.append(
        {
            "fold": fold_idx + 1,
            "subject": test_subject,
            "n_test_samples": len(y_test),
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
        }
    )

    all_y_test.extend(y_test)
    all_y_pred.extend(y_pred)

    print(f"Fold {fold_idx + 1} (Subject {test_subject}):")
    print(f"  Samples: {len(y_test)}")
    print(f"  Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}\n")

# Calculate total time
total_time = time.time() - total_start

# Aggregate results
results_df = pd.DataFrame(fold_results)
overall_cm = confusion_matrix(all_y_test, all_y_pred)

print("\n=== Aggregated Results ===")
print(f"Mean Accuracy: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
print(f"Mean Precision: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
print(f"Mean Recall: {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
print(f"Mean F1-Score: {results_df['f1_score'].mean():.4f} ± {results_df['f1_score'].std():.4f}")
print(f"\nOverall Confusion Matrix (all folds combined):\n{overall_cm}")
print(f"\nTotal LOSO training time: {timedelta(seconds=int(total_time))}")

# Save results
output_data = {
    "model_type": "XGBoost",
    "hyperparameters": best_params,
    "mean_accuracy": float(results_df["accuracy"].mean()),
    "std_accuracy": float(results_df["accuracy"].std()),
    "mean_precision": float(results_df["precision"].mean()),
    "std_precision": float(results_df["precision"].std()),
    "mean_recall": float(results_df["recall"].mean()),
    "std_recall": float(results_df["recall"].std()),
    "mean_f1_score": float(results_df["f1_score"].mean()),
    "std_f1_score": float(results_df["f1_score"].std()),
    "overall_confusion_matrix": overall_cm.tolist(),
    "total_runtime_seconds": int(total_time),
    "per_fold_results": fold_results,
}

output_file = Path(__file__).parent / "XGBoost_fullfeatures_results_lowback.json"
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\nResults saved to {output_file}")
