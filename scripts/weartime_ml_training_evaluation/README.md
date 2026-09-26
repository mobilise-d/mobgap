# Wear-time model training and evaluation

These scripts train the CNN and XGBoost wear-time detectors from raw SUSTAIN CWA
recordings through `SustainWearTimeDataset` and
`WtdEmulationPipeline.self_optimize`. The evaluation scripts split recordings
by day and run participant-grouped cross-validation through TPCP `Optimize`.

Install MobGap with its `weartime` extra. TensorFlow training requires a
supported Python version below 3.13. Supply `--dataset-path` or set
`MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH` to the SUSTAIN wear-time folder.

| Model | Train | Daily evaluation |
| --- | --- | --- |
| CNN | `training/train_cnn.py` | `evaluation/loso_daily_cnn.py` |
| XGBoost | `training/train_xgboost.py` | `evaluation/loso_daily_xgboost.py` |

Run each script with `--help` for its data selection, model, and output
options. The daily evaluation scripts support `--dry-run` to write the dataset
index and fold plan without fitting models.

CNN training saves a `.keras` artifact. Load it in a fresh Python process with
`mobgap.weartime.load_keras_weartime_model(path)` so the optional model-side
standardization layer is registered before deserialization.

The raw SUSTAIN dataset is not distributed with MobGap. Model fitting and daily
cross-validation therefore require access to that dataset.
