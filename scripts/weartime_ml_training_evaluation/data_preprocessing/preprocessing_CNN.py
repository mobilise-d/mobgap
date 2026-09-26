"""
Create windowed training dataset for CNN wear-time detection.

Converts raw IMU data with wear-time labels into windowed format with per-window standardization.
Output: NPZ file with 'X' (windows), 'y' (labels), 'groups' (subject IDs)
"""

import gc
import os
from pathlib import Path

import numpy as np

# Config
RAW_DATA_DIR = os.getenv("WEARTIME_RAW_DATA_DIR")
REFERENCE_FILE = os.getenv("WEARTIME_REFERENCE_FILE")

if RAW_DATA_DIR is None or REFERENCE_FILE is None:
    raise ValueError(
        "Set paths:\n"
        "export WEARTIME_RAW_DATA_DIR=/path/to/raw/data\n"
        "export WEARTIME_REFERENCE_FILE=/path/to/reference.json"
    )

OUTPUT_FILE = Path("windowed_dataset_scaled.npz")
WINDOW_SIZE = 500
OVERLAP = 0.75
# Users may need to adjust to their data, potitional arguments are used, col names do not matter in training
SENSOR_COLS = ["acc_is", "acc_ml", "acc_pa", "gyr_is", "gyr_ml", "gyr_pa"]


def create_windowed_dataset(data, wt_ref, subject_id):
    """Extract overlapping windows with per-window standardization."""
    data_values = data[SENSOR_COLS].values
    step_size = int(WINDOW_SIZE * (1 - OVERLAP))
    n_samples = len(data_values)
    n_windows = (n_samples - WINDOW_SIZE) // step_size + 1

    X = np.zeros((n_windows, WINDOW_SIZE, 6), dtype=np.float32)
    y = np.zeros(n_windows, dtype=np.int32)
    groups = np.full(n_windows, subject_id, dtype=np.int32)

    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + WINDOW_SIZE
        window = data_values[start_idx:end_idx]

        # Standardize per channel within window. This is important, best performance was found with this standardisation
        mean = window.mean(axis=0)
        std = window.std(axis=0)
        std[std < 1e-8] = 1e-8
        X[i] = ((window - mean) / std).astype(np.float32)

        # Label based on window center
        center_idx = start_idx + WINDOW_SIZE // 2
        is_wearing = np.any((wt_ref["start"].values <= center_idx) & (center_idx <= wt_ref["end"].values))
        y[i] = int(is_wearing)

    return X, y, groups


# Load your data here
# Expected format:
# - all_data: dict with structure {subject_id: {sensor_name: {'data': DataFrame}}}
# - Each DataFrame should have columns: ['acc_is', 'acc_ml', 'acc_pa', 'gyr_is', 'gyr_ml', 'gyr_pa']
# - Data should be in body frame, accelerometer in m/s²

raise NotImplementedError("Replace with your data loading code")

# Process data
results = []

for subject_id in all_data.keys():
    for sensor_name, sensor_data in all_data[subject_id].items():
        # Get wear-time reference for this recording
        # Could be DataFrame with ['start', 'end'] as sample indices
        # Example: pd.DataFrame({'start': [0, 10000], 'end': [5000, 20000]})
        wt_ref = ...  # TODO: Extract from your reference data

        try:
            X, y, groups = create_windowed_dataset(data=sensor_data["data"], wt_ref=wt_ref, subject_id=int(subject_id))

            if X.shape[0] > 0:
                results.append({"X": X, "y": y, "groups": groups})
                print(f"Processed {subject_id}: {X.shape[0]} windows")

        except Exception as e:
            print(f"Error {subject_id}: {e}")

# Combine and save
if results:
    X_all = np.concatenate([r["X"] for r in results], axis=0)
    y_all = np.concatenate([r["y"] for r in results], axis=0)
    groups_all = np.concatenate([r["groups"] for r in results], axis=0)

    del results
    gc.collect()

    print(f"\nTotal windows: {len(X_all):,}")
    print(f"Class distribution: {dict(zip(*np.unique(y_all, return_counts=True)))}")

    np.savez(OUTPUT_FILE, X=X_all, y=y_all, groups=groups_all)
    print(f"Saved: {OUTPUT_FILE}")
else:
    print("No data processed")
