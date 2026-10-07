# Wear-time model training and evaluation

The daily LOSO scripts are the training entry points:

- `evaluation/loso_daily_cnn.py`
- `evaluation/loso_daily_xgboost.py`

Edit the configuration constants at the top of each script, then run it with
Python. Install MobGap with its `weartime` extra. TensorFlow requires Python
below 3.13. Set `DATASET_PATH` or `MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH` to the
SUSTAIN wear-time folder. The raw dataset is not distributed with MobGap.

Both scripts use daily datapoints and participant-grouped LOSO through
`WtdEmulationPipeline`, TPCP optimizers, and `EvaluationCV`. Each outer training
fold includes the same seed-42 sample of Part B recordings or days. Held-out
folds contain only human recordings. `OVERLAP` controls the window stride;
set it to `0` for non-overlapping windows.

XGBoost performs Optuna tuning separately within each outer fold. Inner CV
uses human recordings only, with a seeded 40% sample of each inner training
fold's days. The best candidate is refitted on all outer training days,
including the selected Part B data. Recording-level float32 features are
cached with TPCP's hybrid cache and fast best-effort hashing.

CNN trains directly in each outer fold. Windows are prepared lazily without
feature caching. GPU folds run sequentially.

Each run exports only `fold_results.csv`, `daily_results.csv`, and a final
trained model. Fold optimizers are not retained. After LOSO, the optimizer runs
once on the union of all selected human days and the fixed Part B sample.
For XGBoost, this includes a new human-only Optuna search followed by refitting
the best candidate on that complete training set. Its classifier and feature
order are saved as `model.pkl` and `feature_order.pkl`. CNN saves `model.keras`.
These final models are separate from the models used for held-out scoring.

Load CNN `.keras` artifacts in a fresh process with
`mobgap.weartime.load_keras_weartime_model(path)` to register the optional
model-side standardization layer before deserialization.
