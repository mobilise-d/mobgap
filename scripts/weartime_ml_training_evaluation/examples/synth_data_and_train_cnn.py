# %%

"""
End-to-end example: Generate synthetic data and train CNN-LSTM model.

This script demonstrates the complete training workflow using synthetic data.
For production training with real data, use separate scripts in training/ folder.
"""

import json
import time
from datetime import datetime, timedelta

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 70)
print("SYNTHETIC DATA GENERATION + MODEL TRAINING")
print("=" * 70)
# %%
# Step 1: Synthetic Data generation

print("\nStep 1: Generating synthetic data...")

np.random.seed(42)

N_SUBJECTS = 2
N_WINDOWS_PER_SUBJECT = 1000
WINDOW_SIZE = 500
N_CHANNELS = 6

all_windows = []
all_labels = []
all_groups = []

for subject_id in range(1, N_SUBJECTS + 1):
    windows = np.random.randn(N_WINDOWS_PER_SUBJECT, WINDOW_SIZE, N_CHANNELS).astype(np.float32)
    labels = np.random.randint(0, 2, N_WINDOWS_PER_SUBJECT)
    groups = np.full(N_WINDOWS_PER_SUBJECT, subject_id, dtype=np.int32)

    all_windows.append(windows)
    all_labels.append(labels)
    all_groups.append(groups)

X_all = np.concatenate(all_windows, axis=0)
y_all = np.concatenate(all_labels, axis=0)
groups_all = np.concatenate(all_groups, axis=0)

print(f"✓ Generated {len(X_all):,} windows")
print(f"  Shape: {X_all.shape}")
print(f"  Class distribution: {dict(zip(*np.unique(y_all, return_counts=True)))}")


# %%
# Step 2: Train CNN-LSTM model. The script below is similar to the original training script

print("\nStep 2: Training CNN-LSTM model...")

total_start = time.time()

# Hyperparameters
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


def create_cnn_lstm_model(input_shape, params):
    """CNN-LSTM architecture for wear-time detection."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(params["filters"][0], params["kernel_size"], padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling1D(params["pool_size"]),
            layers.Dropout(params["dropout_rate"]),
            layers.Conv1D(params["filters"][1], params["kernel_size"], padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling1D(params["pool_size"]),
            layers.Dropout(params["dropout_rate"]),
            layers.Conv1D(params["filters"][2], params["kernel_size"], padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling1D(params["pool_size"]),
            layers.Dropout(params["dropout_rate"]),
            layers.LSTM(params["lstm_units"], return_sequences=False),
            layers.Dropout(params["dropout_rate"]),
            layers.Dense(params["dense_units"]),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(params["dropout_rate"]),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"], clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


print("Creating model...")
model = create_cnn_lstm_model(X_all.shape[1:], HYPERPARAMETERS)
model.summary()

print(f"\nTraining for {HYPERPARAMETERS['epochs']} epochs...")
training_start = time.time()

history = model.fit(
    X_all,
    y_all,
    batch_size=HYPERPARAMETERS["batch_size"],
    epochs=HYPERPARAMETERS["epochs"],
    verbose=1,
)

training_time = time.time() - training_start

total_time = time.time() - total_start

print("\n" + "=" * 70)
print("TRAINING RESULTS")
print("=" * 70)
print(f"Total time: {timedelta(seconds=int(total_time))}")
print(f"Training time: {timedelta(seconds=int(training_time))}")
print(f"\nFinal epoch ({HYPERPARAMETERS['epochs']}):")
print(f"  Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  Loss: {history.history['loss'][-1]:.4f}")
print("\nFirst epoch:")
print(f"  Accuracy: {history.history['accuracy'][0]:.4f}")
print(f"  Loss: {history.history['loss'][0]:.4f}")
print(f"\nImprovement: {(history.history['accuracy'][-1] - history.history['accuracy'][0]):.4f}")
print("=" * 70)
print("\nDemo complete! For production training, use training/train_cnn_lstm.py")
print("Note: Model not saved (synthetic data demo only)")

print("=" * 70)
print("\nDemo complete! For production training, use training/train_cnn_lstm.py")
print("Note: Model not saved (synthetic data demo only)")

# %%
# Collecting model metadata (similar to MobGap production deployment)

print("\n" + "=" * 70)
print("MODEL METADATA COLLECTION")
print("=" * 70)

metadata = {
    "model_type": "CNN_LSTM",
    "version": "demo",
    "training_date": datetime.now().isoformat(),
    "n_samples_total": len(X_all),
    "input_shape": list(X_all.shape[1:]),
    "hyperparameters": HYPERPARAMETERS,
    "class_distribution": {str(k): int(v) for k, v in zip(*np.unique(y_all, return_counts=True))},
    "final_train_accuracy": float(history.history["accuracy"][-1]),
    "final_train_loss": float(history.history["loss"][-1]),
    "training_time_seconds": int(training_time),
    "tensorflow_version": tf.__version__,
    "numpy_version": np.__version__,
    "notes": "Demo model trained on synthetic data",
}

print("Metadata structure:")
print(json.dumps(metadata, indent=2))

print("\n" + "=" * 70)
print("In production, save this metadata alongside the model for reproducibility")
