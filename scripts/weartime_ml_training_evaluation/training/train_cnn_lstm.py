"""
Train CNN-LSTM model for wear-time detection.

This script trains the production CNN-LSTM model on pre-processed windowed IMU data.
For data preparation, see data/create_training_dataset.py
The loop is presented in a simplified version for illustration purposes with exaclty similar functionality.

Environment variable (optional):
    WEARTIME_DATA_PATH: Path to training data NPZ file
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# HARDWARE CONFIGURATION (adjust based on your system)

# Use GPU if available
# Example:
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     print(f"GPU detected: {gpus}")
#     # Optional: limit GPU memory growth
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)
# else:
#     print("No GPU detected, using CPU")

# Or use CPU-only training
# Example:
# os.environ['TF_NUM_INTRAOP_THREADS'] = '24'
# os.environ['TF_NUM_INTEROP_THREADS'] = '24'
# tf.config.set_visible_devices([], 'GPU')
# tf.config.threading.set_intra_op_parallelism_threads(24)
# tf.config.threading.set_inter_op_parallelism_threads(24)
# print("Configured for CPU training with 24 threads")

print(f"TensorFlow version: {tf.__version__}")
print(f"Using: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}\n")

# Start timing total execution
total_start = time.time()

# CONFIGURATION

# Path to training data (NPZ file with 'X' and 'y' arrays)
# Prepare data using: data/create_training_dataset.py
# Set via environment variable or edit this path:

DATA_PATH = os.getenv("WEARTIME_DATA_PATH")

if DATA_PATH is None:
    raise ValueError(
        "Please set WEARTIME_DATA_PATH environment variable:\n"
        "export WEARTIME_DATA_PATH=/path/to/cnn_windowed_dataset_lowback_scaled_perwindow.npz\n"
        "Or edit DATA_PATH in this script directly."
    )

DATA_PATH = Path(DATA_PATH)

# Output directory for trained models
OUTPUT_DIR = Path("models/production")

# Hyperparameters
# Note: These hyperparameters were selected via nested LOSO cross-validation.
# Epoch count represents the mean stopping epoch across
# all LOSO folds when training with early stopping (max 100 epochs, patience 10).
HYPERPARAMETERS = {
    "num_conv_layers": 3,
    "filters": [32, 64, 128],
    "kernel_size": 9,
    "pool_size": 2,
    "dropout_rate": 0.3,
    "dense_units": 64,
    "learning_rate": 0.001,
    "batch_size": 1024,
    "epochs": 57,
    "lstm_units": 64,
}

# LOAD DATA

print(f"Loading training data from: {DATA_PATH}")
if not Path(DATA_PATH).exists():
    raise FileNotFoundError(
        f"Training data not found: {DATA_PATH}\n"
        f"Please run data/create_training_dataset.py first or set WEARTIME_DATA_PATH"
    )

data = np.load(DATA_PATH)
X_all = data["X"].astype(np.float32)
y_all = data["y"]

print("✓ Data loaded")
print(f"  Total samples: {len(X_all):,}")
print(f"  Input shape: {X_all.shape[1:]}")

unique, counts = np.unique(y_all, return_counts=True)
class_dist = {int(k): int(v) for k, v in zip(unique, counts)}
print(f"  Class distribution: {class_dist}\n")


# MODEL ARCHITECTURE


def create_cnn_lstm_model(input_shape, params):
    """
    Create CNN-LSTM model for wear-time detection.

    Architecture:
    - 3 convolutional blocks (Conv1D + BatchNorm + ReLU + MaxPool + Dropout)
    - LSTM layer for temporal dependencies
    - Dense head with dropout
    - Binary classification output (wear/non-wear)
    """
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            # Conv Block 1
            layers.Conv1D(params["filters"][0], params["kernel_size"], padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling1D(params["pool_size"]),
            layers.Dropout(params["dropout_rate"]),
            # Conv Block 2
            layers.Conv1D(params["filters"][1], params["kernel_size"], padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling1D(params["pool_size"]),
            layers.Dropout(params["dropout_rate"]),
            # Conv Block 3
            layers.Conv1D(params["filters"][2], params["kernel_size"], padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling1D(params["pool_size"]),
            layers.Dropout(params["dropout_rate"]),
            # LSTM for temporal patterns
            layers.LSTM(params["lstm_units"], return_sequences=False),
            layers.Dropout(params["dropout_rate"]),
            # Dense head
            layers.Dense(params["dense_units"]),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(params["dropout_rate"]),
            # Output
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"], clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# TRAINING

print("Creating model...")
model = create_cnn_lstm_model(X_all.shape[1:], HYPERPARAMETERS)
print("✓ Model created")
print("\nModel summary:")
model.summary()
print()

print(f"Training on all {len(X_all):,} samples for {HYPERPARAMETERS['epochs']} epochs...")
sys.stdout.flush()

training_start = time.time()

history = model.fit(
    X_all,
    y_all,
    batch_size=HYPERPARAMETERS["batch_size"],
    epochs=HYPERPARAMETERS["epochs"],
    verbose=1,
)

training_time = time.time() - training_start
print(f"\n✓ Training complete ({timedelta(seconds=int(training_time))})")

# SAVE MODEL AND METADATA

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save model
model_path = OUTPUT_DIR / "cnn_lstm_lowback_model.keras"
model.save(model_path)
print(f"Model saved: {model_path}")

# Save metadata
metadata = {
    "model_type": "CNN_LSTM",
    "version": "production",
    "training_date": datetime.now().isoformat(),
    "n_samples_total": len(X_all),
    "input_shape": list(X_all.shape[1:]),
    "hyperparameters": HYPERPARAMETERS,
    "class_distribution": class_dist,
    "final_train_accuracy": float(history.history["accuracy"][-1]),
    "training_time_seconds": int(training_time),
    "tensorflow_version": tf.__version__,
    "numpy_version": np.__version__,
    "notes": "Production model trained on complete dataset with per-window standardization",
}

metadata_path = OUTPUT_DIR / "cnn_lstm_lowback_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved: {metadata_path}")

# SUMMARY

total_time = time.time() - total_start

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Total time: {timedelta(seconds=int(total_time))}")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Model: {model_path}")
print(f"Metadata: {metadata_path}")
print("=" * 70)
