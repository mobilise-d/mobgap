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
options. The daily evaluation scripts support `--dry-run` to write the fold
plan without fitting models.

The XGBoost scripts accept `--overlap` (default `0.75`; `0` uses non-overlapping
windows). They cache one float32 feature result per selected recording or day
under `--cache-dir/xgboost_features`, shared between training and evaluation
scoring. TPCP's fast best-effort hash is used for cache lookups; changing this option
invalidates entries created with the default hash.

XGBoost daily evaluation scores held-out days only. It runs a separate
Optuna search in every outer participant fold. Each trial uses participant-grouped
inner cross-validation and samples 40% of each inner training fold's human day
rows. Inner folds use human recordings only. The LOSO script supplies the inner
splitter to the Optuna optimizer. The best parameters are refit on every outer
training day, including all selected part B days, before scoring the held-out
participant. Use `--n-trials`, `--inner-folds`,
`--search-train-fraction`, and `--search-seed` to adjust the search.
`optuna_best_by_fold.csv` and `optuna_trials.csv` record the search results.
Both daily evaluations write `timings.json` from `EvaluationCV.perf_`.

Both daily evaluation scripts use TPCP's `CombinedSplitter` and `NoSplit` to put
the same sampled part B recordings in training for every fold. Test folds contain
only part A human recordings. By default, the scripts sample two part B recordings
with seed 42; `--part-b-recording-count` changes the count. The splitter applies
both the human participant selection and the fixed part B training selection.
The dry run writes the fold plan to `fold_metadata.csv`.

CNN training saves a `.keras` artifact. Load it in a fresh Python process with
`mobgap.weartime.load_keras_weartime_model(path)` so the optional model-side
standardization layer is registered before deserialization.

The raw SUSTAIN dataset is not distributed with MobGap. Model fitting and daily
cross-validation therefore require access to that dataset.
