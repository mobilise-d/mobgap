# Wear-Time Detection Model Training

Training scripts and reproducibility materials for IMU-based wear-time detection algorithms developed as part of the SUSTAIN/EFPIA project.

## Overview

This repository contains everything needed to evaluate and retrain the CNN wear-time detection models used in MobGap when dependencies are updated or improvements are made.

**Models included:**
- CNN (pure convolutional)
- CNN-LSTM (convolutional + LSTM for temporal patterns)

Both models use per-window standardized 5-second IMU windows (accelerometer + gyroscope) to classify wear vs non-wear periods.

## Evaluation

Each training run automatically evaluates the model using LOSO cross-validation and saves:

- Per-subject JSON files with predictions, probabilities, and metrics
- An aggregated `loso_summary.json` with mean ± std across all folds
- Diagnostic plots: ROC curve, Precision-Recall curve, calibration curve, and loss/accuracy curves (mean ± std across folds)

## Quick Start

**Demo with synthetic data:**

See `examples/synthetic_training_demo.py` for a complete end-to-end example using synthetic data.

This generates synthetic data and trains a model end-to-end. Good for testing the training pipeline without real IMU data.

## Training on Real Data

### Step 1: Prepare Your Data

Your data needs to be:
- Body-frame IMU data with columns: `acc_is`, `acc_ml`, `acc_pa`, `gyr_is`, `gyr_ml`, `gyr_pa`
- Accelerometer in m/s²
- Wear-time labels as sample index intervals or timestamps

Adapt `data/create_training_dataset.py` to your data format. The script shows the required windowing and per-window standardization procedure.

**Output:** NPZ file with:
- `X`: windowed data, shape (n_windows, 500, 6)
- `y`: labels, 0=non-wear, 1=wear
- `groups`: subject IDs for each window

The script applies per-window standardization (zero mean, unit std per channel).

### Step 2: Train Models

**CNN-LSTM (recommended, slightly better accuracy):**

Configure and run `training/train_cnn_lstm.py`

**Pure CNN (faster training, similar performance):**

Configure and run `training/train_cnn.py`

Both scripts expect `WEARTIME_DATA_PATH` environment variable or direct path editing in the script. Models and metadata saved in `models/production/`

## Requirements

```
tensorflow>=2.18
numpy
pandas
```

See `requirements.txt` for exact versions used in original training.

## Model Details

Hyperparameters were optimized via 26-fold Leave-One-Subject-Out cross-validation on 26 subjects (~6.5M windows). Epoch counts represent mean early-stopping points across folds.

Full methodology in technical reports (SUSTAIN consortium members have access).

## Hardware

Training tested on:
- CPU: 24-core machine, ~23 hours for CNN-LSTM
- GPU: Much faster (hours instead of days)

Both scripts work with CPU or GPU. For GPU, uncomment the GPU configuration section.
