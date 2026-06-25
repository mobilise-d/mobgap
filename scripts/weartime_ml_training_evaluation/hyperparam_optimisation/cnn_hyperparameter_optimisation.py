"""
1D CNN Hyperparameter Optimisation using Nested LOSO Cross-Validation
======================================================================

This script performs hyperparameter optimisation for the 1D CNN wear-time
detection model using a nested Leave-One-Subject-Out (LOSO) cross-validation
framework.

Two-stage procedure:
    1. LOSO evaluation: For each outer fold, one subject is held out as the
       test set. Hyperparameters are selected independently within each fold
       using an inner 2-fold stratified CV on the remaining training subjects.
       The reported test accuracy per fold is therefore fully independent of
       hyperparameter selection — no data leakage.

    2. Deployment configuration: After all folds complete, a single best
       hyperparameter configuration is selected using frequency-weighted
       scoring across folds. This is used only for the final deployed model
       and has no influence on the reported LOSO evaluation results.

The best configuration is selected using frequency-weighted scoring: for each unique hyperparameter set,
the product of selection frequency across folds and mean inner CV accuracy is computed (frequency × mean_CV).
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import json
import gc
import time
from datetime import timedelta
from collections import defaultdict
import sys

import tensorflow as tf

# ── Metal GPU setup ───────────────────────────────────────────
physical_devices = tf.config.list_physical_devices()
print(f"[DEBUG] Available devices: {physical_devices}")
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"[DEBUG] Metal GPU detected: {gpu_devices}")
else:
    print("[DEBUG] No GPU detected - running on CPU")
print(f"[DEBUG] TensorFlow version: {tf.__version__}")

# ── Data path ─────────────────────────────────────────────────
data_path = Path("cnn_windowed_dataset.npz")

if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

# ── Load data into RAM ────────────────────────────────────────
print("\n" + "=" * 60)
print("LOADING DATA INTO RAM")
print("=" * 60)

data = np.load(data_path)
X_combined = data['X'].astype(np.float32)
groups_combined = data['groups']
labels_combined = data['y']

input_shape = X_combined.shape[1:]
n_samples_a = len(groups_combined)

print(f"\n✓ Data loaded: {X_combined.shape}, {X_combined.nbytes / 1024 ** 3:.1f} GB")
print(f"  Subjects: {len(np.unique(groups_combined))}")
print("=" * 60 + "\n")
sys.stdout.flush()

# ── Resume from checkpoint if available ────────────────────
resume_file = Path("cnn_hyperopt_results.json")
completed_folds = []
cv_results = []
previous_runtime = 0

if resume_file.exists():
    print(f"\n[RESUME] Found existing results at {resume_file}")
    with open(resume_file, 'r') as f:
        previous_results = json.load(f)
        completed_folds = [r['subject'] for r in previous_results['fold_results']]
        cv_results = previous_results['fold_results']
        if 'total_runtime_seconds' in previous_results and previous_results['total_runtime_seconds']:
            previous_runtime = previous_results['total_runtime_seconds']
    print(f"[RESUME] Skipping {len(completed_folds)} completed subjects: {completed_folds}")
    print(f"[RESUME] Previous runtime: {timedelta(seconds=previous_runtime)}")
    print("=" * 60 + "\n")


# ── CNN architecture (matching final model) ───────────────────
def create_1d_cnn(input_shape, num_conv_layers, filters, kernel_size, pool_size,
                  dropout_rate, dense_units, learning_rate):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    # First conv layer
    model.add(layers.Conv1D(filters[0], kernel_size, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size))
    model.add(layers.Dropout(dropout_rate))

    # Additional conv layers
    for i in range(1, num_conv_layers):
        model.add(layers.Conv1D(filters[i], kernel_size, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling1D(pool_size))
        model.add(layers.Dropout(dropout_rate))

    # Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Hyperparameter grid ───────────────────────────────────────
param_grid = [
    {'num_conv_layers': 2, 'filters': [32, 64], 'kernel_size': 7, 'pool_size': 2, 'dropout_rate': 0.2,
     'dense_units': 64, 'learning_rate': 0.001, 'batch_size': 1024, 'epochs': 100},
    {'num_conv_layers': 2, 'filters': [32, 64], 'kernel_size': 9, 'pool_size': 2, 'dropout_rate': 0.3,
     'dense_units': 64, 'learning_rate': 0.001, 'batch_size': 1024, 'epochs': 100},
    {'num_conv_layers': 3, 'filters': [32, 64, 128], 'kernel_size': 7, 'pool_size': 2, 'dropout_rate': 0.2,
     'dense_units': 64, 'learning_rate': 0.001, 'batch_size': 1024, 'epochs': 100},
    {'num_conv_layers': 3, 'filters': [32, 64, 128], 'kernel_size': 9, 'pool_size': 2, 'dropout_rate': 0.3,
     'dense_units': 64, 'learning_rate': 0.001, 'batch_size': 1024, 'epochs': 100},
    {'num_conv_layers': 3, 'filters': [32, 64, 128], 'kernel_size': 9, 'pool_size': 2, 'dropout_rate': 0.3,
     'dense_units': 128, 'learning_rate': 0.001, 'batch_size': 1024, 'epochs': 100},
]


def train_single_combination(params, train_idx, n_samples_a, n_inner_folds=2):
    """Train a single hyperparameter combination with inner CV."""
    accuracy_scores = []

    train_labels = labels_combined[train_idx]
    inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=42)

    for inner_fold, (tr_idx, val_idx) in enumerate(inner_cv.split(train_idx, train_labels)):
        actual_tr_idx = train_idx[tr_idx]
        actual_val_idx = train_idx[val_idx]

        X_tr = X_combined[actual_tr_idx]
        y_tr = labels_combined[actual_tr_idx]
        X_val = X_combined[actual_val_idx]
        y_val = labels_combined[actual_val_idx]

        model = create_1d_cnn(
            input_shape=input_shape,
            num_conv_layers=params['num_conv_layers'],
            filters=params['filters'],
            kernel_size=params['kernel_size'],
            pool_size=params['pool_size'],
            dropout_rate=params['dropout_rate'],
            dense_units=params['dense_units'],
            learning_rate=params['learning_rate']
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=0
        )

        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            batch_size=params['batch_size'],
            epochs=50,  # reduced for inner CV only
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        y_pred = (model.predict(X_val, batch_size=params['batch_size'], verbose=0) > 0.5).astype(int).flatten()
        accuracy_scores.append(accuracy_score(y_val, y_pred))

        del model
        K.clear_session()
        gc.collect()

    return np.mean(accuracy_scores)


# ── LOSO cross-validation with grid search ────────────────────
gkf = GroupKFold(n_splits=len(np.unique(groups_combined)))
start_time = time.time()

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(labels_combined, labels_combined, groups_combined)):
    fold_start = time.time()
    test_subject = str(groups_combined[test_idx][0]).zfill(3)

    # Skip if already completed
    if test_subject in completed_folds:
        print(f"\n[SKIP] Fold {fold_idx + 1} (Subject {test_subject}) - already completed")
        continue

    print(f"\n{'=' * 60}")
    print(f"FOLD {fold_idx + 1}/{len(np.unique(groups_combined))} — Subject {test_subject}")
    print(f"{'=' * 60}")
    print(f"  Train: {len(train_idx):,}, Test: {len(test_idx):,}")
    sys.stdout.flush()

    # Grid search
    results = []
    for param_idx, params in enumerate(param_grid):
        print(f"  Config {param_idx + 1}/{len(param_grid)}...", end=" ", flush=True)
        avg_acc = train_single_combination(params, train_idx, n_samples_a)
        results.append((avg_acc, params))
        print(f"Inner CV Accuracy: {avg_acc:.4f}")

    best_accuracy, best_params = max(results, key=lambda x: x[0])
    print(f"\n  Best inner CV accuracy: {best_accuracy:.4f}")
    print(f"  Best params: {best_params}")

    # Refit on full training data with val split
    np.random.seed(42)
    shuffled = np.random.permutation(train_idx)
    val_split = int(0.9 * len(shuffled))
    actual_train_idx = shuffled[:val_split]
    actual_val_idx = shuffled[val_split:]

    X_train = X_combined[actual_train_idx]
    y_train = labels_combined[actual_train_idx]
    X_val = X_combined[actual_val_idx]
    y_val = labels_combined[actual_val_idx]
    X_test = X_combined[test_idx]
    y_test = labels_combined[test_idx]

    final_model = create_1d_cnn(
        input_shape=input_shape,
        num_conv_layers=best_params['num_conv_layers'],
        filters=best_params['filters'],
        kernel_size=best_params['kernel_size'],
        pool_size=best_params['pool_size'],
        dropout_rate=best_params['dropout_rate'],
        dense_units=best_params['dense_units'],
        learning_rate=best_params['learning_rate']
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    print(f"\n  Refitting best model on full training set...")
    final_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=best_params['batch_size'],
        epochs=best_params['epochs'],
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    y_pred = (final_model.predict(X_test, batch_size=best_params['batch_size'], verbose=0) > 0.5).astype(int).flatten()
    test_acc = accuracy_score(y_test, y_pred)

    cv_results.append({
        'fold': fold_idx + 1,
        'subject': test_subject,
        'best_params': best_params,
        'best_cv_score': float(best_accuracy),
        'test_acc': float(test_acc)
    })

    # Incremental save after each fold
    temp_output = {
        'mean_test_accuracy': None,  # Will be calculated at the end
        'std_test_accuracy': None,
        'best_overall_params': None,
        'best_overall_cv_accuracy': None,
        'total_runtime_seconds': int(time.time() - start_time) + previous_runtime,
        'fold_results': cv_results
    }
    with open(resume_file, 'w') as f:
        json.dump(temp_output, f, indent=2)
    print(f"  [SAVED] Progress checkpoint saved to {resume_file}")

    fold_time = time.time() - fold_start
    print(f"\n  Test Accuracy: {test_acc:.4f}, Fold time: {timedelta(seconds=int(fold_time))}")
    sys.stdout.flush()

    del final_model, X_train, y_train, X_val, y_val, X_test
    K.clear_session()
    gc.collect()

# ── Aggregate results ─────────────────────────────────────────
total_time = time.time() - start_time
mean_acc = np.mean([r['test_acc'] for r in cv_results])
std_acc = np.std([r['test_acc'] for r in cv_results])

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
for r in cv_results:
    print(f"  Fold {r['fold']} ({r['subject']}): CV={r['best_cv_score']:.4f}, Test={r['test_acc']:.4f}")
print(f"\nMean Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
print(f"Total runtime: {timedelta(seconds=int(total_time))}")

# Select best overall hyperparameters with frequency-weighted scoring
param_cv_scores = defaultdict(list)
for r in cv_results:
    key = json.dumps(r['best_params'], sort_keys=True, default=str)
    param_cv_scores[key].append(r['best_cv_score'])

# Frequency-weighted CV score selection
param_stats = {}
for key, cv_scores in param_cv_scores.items():
    frequency = len(cv_scores)
    mean_cv = np.mean(cv_scores)

    # Require minimum 2 folds for consideration
    if frequency >= 2:
        # Weight: favor both frequency and performance
        score = frequency * mean_cv  # Option 1: Multiply
        # score = 0.3 * (frequency / len(cv_results)) + 0.7 * mean_cv  # Option 2: Weighted average

        param_stats[key] = {
            'frequency': frequency,
            'mean_cv': mean_cv,
            'weighted_score': score
        }

if param_stats:
    best_key = max(param_stats, key=lambda k: param_stats[k]['weighted_score'])
else:
    # Fallback to most frequent
    best_key = max(param_cv_scores, key=lambda k: len(param_cv_scores[k]))

best_overall_params = json.loads(best_key)
best_overall_cv_acc = np.mean(param_cv_scores[best_key])

# Print selection details
print(f"\nHyperparameter Selection Details:")

# First, collect test accuracies per config
param_test_scores = defaultdict(list)
for r in cv_results:
    key = json.dumps(r['best_params'], sort_keys=True, default=str)
    param_test_scores[key].append(r['test_acc'])

for key in param_cv_scores.keys():
    params_display = json.loads(key)
    freq = len(param_cv_scores[key])
    cv = np.mean(param_cv_scores[key])
    test = np.mean(param_test_scores[key])
    print(f"  Config (k={params_display['kernel_size']}, d={params_display['dropout_rate']}, "
          f"dense={params_display['dense_units']}): "
          f"freq={freq}/{len(cv_results)}, mean_CV={cv:.4f}, mean_Test={test:.4f}")

print(f"\nBest overall params: {best_overall_params}")
print(f"Mean inner CV accuracy: {best_overall_cv_acc:.4f}")

# Save results
output_data = {
    'mean_test_accuracy': float(mean_acc),
    'std_test_accuracy': float(std_acc),
    'best_overall_params': best_overall_params,
    'best_overall_cv_accuracy': float(best_overall_cv_acc),
    'total_runtime_seconds': int(total_time),
    'fold_results': cv_results
}

output_file = Path("cnn_hyperopt_results.json")
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"\nResults saved to {output_file}")