"""
Train final XGBoost model for production deployment.
Saves model as .pkl and meta data as .json.

Environment variable (optional):
    WEARTIME_DATA_PATH: Path to training data JSON file

Note: The feature order used during preprocessing and retraining does not need to match
the feature order of the production model. At inference, WtdMegaritisXGBoost automatically
reorders features to match the saved feature order file before prediction.
"""

import json
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path

import xgboost as xgb

from scripts.weartime_ml_training_evaluation.training.utils import load_json_to_dataframe_flat

# Start timing
total_start = time.time()

# CONFIG
DATA_PATH = os.getenv("WEARTIME_DATA_PATH")

if DATA_PATH is None:
    raise ValueError(
        "Please set WEARTIME_DATA_PATH environment variable:\n"
        "export WEARTIME_DATA_PATH=/path/to/features_xgboost.json\n"
        "Or edit DATA_PATH in this script directly."
    )

# Output directory for trained models
OUTPUT_DIR = Path("models/production")

# Load ALL data (no train/test split)
data = load_json_to_dataframe_flat(DATA_PATH)

# Separate features and target
features = data.drop(columns=["id", "sensor", "segment_id", "label", "window_center_idx"], errors="ignore")
target = data["label"]

print("=== Training Final Production Model (XGBoost Full Features) ===")
print(f"Total samples: {len(features):,}")
print(f"Total features: {len(features.columns)}")
print(f"Class distribution: {target.value_counts().to_dict()}\n")

# Best parameters from LOSO evaluation
best_params = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "random_state": 42,
}

# Train model on ALL data (no scaling needed for XGBoost)
model = xgb.XGBClassifier(**best_params)
model.fit(features, target)

# Calculate total time
total_time = time.time() - total_start

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save model
model_path = OUTPUT_DIR / "xgboost_fullfeatures_lowback_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"✓ Model saved: {model_path}")

# Save metadata
metadata = {
    "model_type": "XGBoost",
    "version": "fullfeatures",
    "training_date": datetime.now().isoformat(),
    "n_samples": len(features),
    "n_features": len(features.columns),
    "feature_names": features.columns.tolist(),
    "hyperparameters": best_params,
    "class_distribution": target.value_counts().to_dict(),
    "training_time_seconds": int(total_time),
    "xgboost_version": xgb.__version__,
    "notes": "Trained on all 26 subjects. No feature scaling required for tree-based model.",
}

metadata_path = OUTPUT_DIR / "xgboost_fullfeatures_lowback_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved: {metadata_path}")

# Save feature order
feature_order_path = OUTPUT_DIR / "xgboost_fullfeatures_lowback_feature_order.pkl"
with open(feature_order_path, "wb") as f:
    pickle.dump(features.columns.tolist(), f)
print(f"Feature order saved here: {feature_order_path}")

print(f"Total training time: {timedelta(seconds=int(total_time))}")
print(f"Files saved in: {OUTPUT_DIR}")
