# Wear-time model training and evaluation

The daily LOSO scripts are the training entry points:

- `evaluation/loso_daily_cnn.py`
- `evaluation/loso_daily_xgboost.py`

Edit the configuration constants at the top of each script, then run it with
Python. Install MobGap with its `weartime` extra. TensorFlow requires Python
below 3.13. Set `DATASET_PATH` or `MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH` to the
SUSTAIN wear-time folder. The raw dataset is not distributed with MobGap.

Both scripts use daily datapoints and participant-grouped LOSO through
`WtdEmulationPipeline`, TPCP optimizers, and `EvaluationCV`. Each human participant's
entire set of selected days is held out together. Part B days use one seed-42
permutation, split into disjoint halves. The same training half and test half are
included in every outer fold. With an odd day count the test half has one extra
day. Part B test days therefore repeat across folds; they are not independent
additional held-out participants. `OVERLAP` controls the window stride; set it
to `0` for non-overlapping windows.

XGBoost performs Optuna tuning separately within each outer fold. Inner CV
holds out human participants with GroupKFold. Inner validation and candidate
ranking contain human days only. The optimizer samples 40% of each inner human
training set. `PART_B_DAY_COUNT=None` keeps all provided Part B training days unchanged; an
integer independently samples that many seeded Part B days from the provided
outer training half in the training dataset transform. The best candidate is
refitted on the full outer training fold, including its complete Part B training
half. Recording-level float32 features use TPCP's hybrid cache.

CNN trains directly in each outer fold. Windows are prepared lazily without
feature caching. GPU folds run sequentially.

Each run exports only `fold_results.csv`, `daily_results.csv`, and a final
trained model. Fold optimizers are not retained. After LOSO, the optimizer runs
once on the union of all selected human days and both Part B halves. This final
artifact fit happens after scoring and does not change the outer evaluation.
For XGBoost, this includes a new search with human-only inner validation followed by refitting
the best candidate on that complete training set. Its classifier and feature
order are saved as `model.pkl` and `feature_order.pkl`. CNN saves `model.keras`.
These final models are separate from the models used for held-out scoring.

Load CNN `.keras` artifacts in a fresh process with
`mobgap.weartime.load_keras_weartime_model(path)` to register the optional
model-side standardization layer before deserialization.
