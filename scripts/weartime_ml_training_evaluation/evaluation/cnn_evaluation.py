"""
1D CNN training and evaluation using Leave-One-Subject-Out (LOSO) cross-validation.
This script is added for reproducibility and future retraining of the CNN wear-time model.

Loads windowed IMU data from NPZ files, trains a 1D CNN per fold, and evaluates
using standard binary classification metrics. Saves per-fold JSON results, aggregated
results, training histories, and diagnostic plots (ROC, PR, calibration, loss, accuracy).

Expected NPZ format
-------------------
Each NPZ file must contain:
    X       : float array, shape (n_samples, window_length, n_channels)
    y       : int array,   shape (n_samples,)   — binary labels (0/1)
    groups  : array,       shape (n_samples,)   — subject identifiers

"""

import gc
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers

print(f"TensorFlow version : {tf.__version__}")
sys.stdout.flush()


# Edit these paths and hyperparameters before running
# ── Choose one configuration and comment out the other ───────────────────────

# CNN only
CONFIG = {
    "data_path": "/path/to/your/dataset.npz",
    "output_dir": "./loso_outputs_cnn",
    "plot_dir": "./loso_plots_cnn",
    "hyperparameters": {
        "num_conv_layers": 3,
        "filters": [32, 64, 128],
        "kernel_size": 9,
        "pool_size": 2,
        "dropout_rate": 0.3,
        "dense_units": 64,
        "learning_rate": 0.001,
        "batch_size": 1024,
        "epochs": 100,
        "lstm_units": 0,
    },
    # Fraction of training data reserved for validation (per fold).
    "val_split": 0.1,
    # Random seed for reproducibility.
    "random_seed": 42,
}

# CNN-LSTM
# CONFIG = {
#     "data_path": "/path/to/your/dataset.npz",
#     "output_dir": "./loso_outputs_cnn_lstm",
#     "plot_dir": "./loso_plots_cnn_lstm",
#     "hyperparameters": {
#         "num_conv_layers": 3,
#         "filters": [32, 64, 128],
#         "kernel_size": 9,
#         "pool_size": 2,
#         "dropout_rate": 0.3,
#         "dense_units": 64,
#         "learning_rate": 0.001,
#         "batch_size": 1024,
#         "epochs": 100,
#         "lstm_units": 64,
#     },
#     "val_split": 0.1,
#     "random_seed": 42,
# }

# ── Model definition ──────────────────────────────────────────────────────────


def create_1d_cnn(
    input_shape,
    num_conv_layers,
    filters,
    kernel_size,
    pool_size,
    dropout_rate,
    dense_units,
    learning_rate,
):
    """
    Build and compile a 1D CNN for binary IMU classification.

    Parameters
    ----------
    input_shape     : tuple  — (window_length, n_channels)
    num_conv_layers : int    — number of Conv1D blocks
    filters         : list   — number of filters per Conv1D block
    kernel_size     : int    — convolution kernel size
    pool_size       : int    — max-pooling stride
    dropout_rate    : float  — dropout probability after each block
    dense_units     : int    — units in the penultimate dense layer
    learning_rate   : float  — Adam learning rate

    Returns
    -------
    keras.Sequential model (compiled)
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    for i in range(num_conv_layers):
        model.add(layers.Conv1D(filters[i], kernel_size, padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling1D(pool_size))
        model.add(layers.Dropout(dropout_rate))

    # Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Data loading ──────────────────────────────────────────────────────────────


def load_npz_file(path):
    """
    Load a single NPZ dataset file.

    The file must contain arrays keyed 'X', 'y', and 'groups'.

    Parameters
    ----------
    path : str or Path — path to the NPZ file

    Returns
    -------
    X      : float32 ndarray, shape (N, window_length, channels)
    y      : ndarray, shape (N,)
    groups : ndarray, shape (N,)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    print(f"  Loading {p.name} ...", end=" ", flush=True)
    data = np.load(p)
    X = data["X"].astype(np.float32)
    y = data["y"]
    groups = data["groups"]
    print(f"{X.shape}  ({X.nbytes / 1024**3:.2f} GB)")

    return X, y, groups


# ── Plot generation ───────────────────────────────────────────────────────────


def save_plots(all_y_true, all_y_prob, all_histories, plot_dir):
    """
    Generate and save five diagnostic plots.

    Plots saved
    -----------
    calibration_curve.png
    precision_recall_curve.png
    roc_curve.png
    loss_curves.png          (mean ± std across folds)
    accuracy_curves.png      (mean ± std across folds)
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)

    # 1. Calibration curve
    fig, ax = plt.subplots(figsize=(6, 6))
    frac_pos, mean_pred = calibration_curve(all_y_true, all_y_prob, n_bins=10)
    ax.plot(mean_pred, frac_pos, "o-", color="steelblue", label="CNN")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Precision-Recall curve
    fig, ax = plt.subplots(figsize=(6, 6))
    prec, rec, _ = precision_recall_curve(all_y_true, all_y_prob)
    ap = average_precision_score(all_y_true, all_y_prob)
    ax.plot(rec, prec, color="steelblue", lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. ROC curve
    fig, ax = plt.subplots(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4 & 5. Loss and accuracy curves (mean ± std across folds)
    if all_histories:
        max_epochs = max(h["epochs_trained"] for h in all_histories)

        def pad(values, length):
            return values + [values[-1]] * (length - len(values))

        train_loss = np.array([pad(h["loss"], max_epochs) for h in all_histories])
        val_loss = np.array([pad(h["val_loss"], max_epochs) for h in all_histories])
        train_acc = np.array([pad(h["accuracy"], max_epochs) for h in all_histories])
        val_acc = np.array([pad(h["val_accuracy"], max_epochs) for h in all_histories])
        epochs = np.arange(1, max_epochs + 1)

        for metric_train, metric_val, ylabel, filename in [
            (train_loss, val_loss, "Binary Cross-Entropy Loss", "loss_curves.png"),
            (train_acc, val_acc, "Accuracy", "accuracy_curves.png"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for values, colour, label in [
                (metric_train, "steelblue", f"Training {ylabel.split()[0].lower()}"),
                (metric_val, "tomato", f"Validation {ylabel.split()[0].lower()}"),
            ]:
                mean, std = values.mean(axis=0), values.std(axis=0)
                ax.plot(epochs, mean, color=colour, lw=2, label=label.capitalize())
                ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=colour)
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_dir / filename, dpi=150, bbox_inches="tight")
            plt.close()

    saved = [
        "calibration_curve.png",
        "precision_recall_curve.png",
        "roc_curve.png",
        "loss_curves.png",
        "accuracy_curves.png",
    ]
    print("\nPlots saved:")
    for name in saved:
        print(f"  {plot_dir / name}")


# ── Main LOSO loop ────────────────────────────────────────────────────────────


def main():
    cfg = CONFIG
    hp = cfg["hyperparameters"]
    np.random.seed(cfg["random_seed"])

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    t0 = time.time()
    X, y, groups = load_npz_file(cfg["data_path"])
    print(f"\n✓ Loaded in {timedelta(seconds=int(time.time() - t0))}")
    print(f"  Shape  : {X.shape}")
    print(f"  Memory : {X.nbytes / 1024**3:.2f} GB")
    print(f"  Subjects: {len(np.unique(groups))}")
    print("=" * 60 + "\n")
    sys.stdout.flush()

    input_shape = X.shape[1:]
    unique_subjects = np.unique(groups)
    n_folds = len(unique_subjects)
    gkf = GroupKFold(n_splits=n_folds)

    fold_results = []
    all_y_test, all_y_pred, all_y_prob = [], [], []
    all_histories = []

    print(f"STARTING LOSO CROSS-VALIDATION ({n_folds} folds)")
    print("=" * 60 + "\n")
    total_start = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(y, y, groups)):
        fold_start = time.time()
        test_subject = str(groups[test_idx][0])

        print(f"\n{'=' * 60}")
        print(f"FOLD {fold_idx + 1}/{n_folds}  —  test subject: {test_subject}")
        print(f"{'=' * 60}")

        # ── Check for existing result ─────────────────────────────────────────
        result_file = output_dir / f"subject_{test_subject}.json"
        if result_file.exists():
            print(f"  Already completed — loading {result_file.name}")
            with open(result_file) as f:
                saved = json.load(f)
            if "predictions" in saved:
                all_y_test.extend(saved["predictions"]["y_true"])
                all_y_pred.extend(saved["predictions"]["y_pred"])
                all_y_prob.extend(saved["predictions"]["y_prob"])
            if "training_history" in saved:
                all_histories.append(saved["training_history"])
            continue

        # ── Train / val split ─────────────────────────────────────────────────
        perm = np.random.permutation(len(train_idx))
        val_cut = int((1 - cfg["val_split"]) * len(train_idx))
        tr_idx = train_idx[perm[:val_cut]]
        val_idx = train_idx[perm[val_cut:]]

        X_train, y_train = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        print(f"  Train: {len(tr_idx):,}  |  Val: {len(val_idx):,}  |  Test: {len(test_idx):,}")
        sys.stdout.flush()

        # ── Build & train model ───────────────────────────────────────────────
        model = create_1d_cnn(
            input_shape=input_shape,
            num_conv_layers=hp["num_conv_layers"],
            filters=hp["filters"],
            kernel_size=hp["kernel_size"],
            pool_size=hp["pool_size"],
            dropout_rate=hp["dropout_rate"],
            dense_units=hp["dense_units"],
            learning_rate=hp["learning_rate"],
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        history = model.fit(
            X_train,
            y_train,
            batch_size=hp["batch_size"],
            epochs=hp["epochs"],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )

        n_epochs = len(history.history["loss"])
        print(f"  Training complete — stopped at epoch {n_epochs}")

        history_dict = {
            "fold": fold_idx + 1,
            "subject": test_subject,
            "loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
            "accuracy": history.history["accuracy"],
            "val_accuracy": history.history["val_accuracy"],
            "epochs_trained": n_epochs,
        }
        all_histories.append(history_dict)

        # ── Evaluate ──────────────────────────────────────────────────────────
        y_prob = model.predict(X_test, batch_size=hp["batch_size"], verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n  Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
        print(f"  Confusion matrix:\n{cm}")

        fold_results.append(
            {
                "fold": fold_idx + 1,
                "subject": test_subject,
                "n_test_samples": len(y_test),
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "confusion_matrix": cm.tolist(),
            }
        )

        all_y_test.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(y_prob.tolist())

        # ── Save per-fold result ──────────────────────────────────────────────
        fold_output = {
            "subject": test_subject,
            "hyperparameters": hp,
            "n_test_samples": len(y_test),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "predictions": {
                "y_true": y_test.tolist(),
                "y_pred": y_pred.tolist(),
                "y_prob": y_prob.tolist(),
            },
            "training_history": history_dict,
        }
        with open(result_file, "w") as f:
            json.dump(fold_output, f, indent=2)
        print(f"  Saved: {result_file}")

        # ── Timing ────────────────────────────────────────────────────────────
        elapsed = time.time() - total_start
        avg_fold = elapsed / (fold_idx + 1)
        remaining = avg_fold * (n_folds - fold_idx - 1)
        print(f"\n  Fold time : {timedelta(seconds=int(time.time() - fold_start))}")
        print(f"  Elapsed   : {timedelta(seconds=int(elapsed))}")
        print(f"  Remaining : {timedelta(seconds=int(remaining))}")
        sys.stdout.flush()

        # ── Cleanup ───────────────────────────────────────────────────────────
        del model, X_train, y_train, X_val, y_val, X_test, y_test, y_pred
        K.clear_session()
        gc.collect()

    # ── Aggregate results ─────────────────────────────────────────────────────
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)

    if fold_results:
        df = pd.DataFrame(fold_results)
        overall_cm = confusion_matrix(all_y_test, all_y_pred)
        for metric in ("accuracy", "precision", "recall", "f1_score"):
            print(f"  {metric.capitalize():12s}: {df[metric].mean():.4f} ± {df[metric].std():.4f}")
        print(f"\nOverall confusion matrix:\n{overall_cm}")

        summary = {
            "model_type": "CNN_1D_LOSO",
            "hyperparameters": hp,
            "mean_accuracy": float(df["accuracy"].mean()),
            "std_accuracy": float(df["accuracy"].std()),
            "mean_precision": float(df["precision"].mean()),
            "std_precision": float(df["precision"].std()),
            "mean_recall": float(df["recall"].mean()),
            "std_recall": float(df["recall"].std()),
            "mean_f1_score": float(df["f1_score"].mean()),
            "std_f1_score": float(df["f1_score"].std()),
            "overall_confusion_matrix": overall_cm.tolist(),
            "total_runtime_seconds": int(total_time),
            "per_fold_results": fold_results,
        }
        results_file = output_dir / "loso_summary.json"
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {results_file}")

        history_file = output_dir / "training_histories.json"
        with open(history_file, "w") as f:
            json.dump(all_histories, f, indent=2)
        print(f"Histories saved to {history_file}")

    else:
        print("All folds were already completed; check per-subject JSON files.")

    print(f"\nTotal runtime: {timedelta(seconds=int(total_time))}")
    print("=" * 60)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if all_y_test:
        save_plots(all_y_test, all_y_prob, all_histories, cfg["plot_dir"])
    else:
        print("\nNo prediction data available for plotting.")


if __name__ == "__main__":
    main()
