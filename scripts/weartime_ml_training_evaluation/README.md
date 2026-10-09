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
entire set of selected days is held out together. Simulated non-wear source
recordings use one seed-42 permutation, split into disjoint halves. The same
training half and test half are included in every outer fold. With an odd
recording count the test half has one extra recording. All its days remain in
the same half. Simulated non-wear test days therefore repeat across folds; they
are not independent
additional held-out participants. `OVERLAP` controls the window stride; set it
to `0` for non-overlapping windows.

XGBoost uses the reusable `mobgap.weartime.optimization.WearTimeOptunaSearch`
and `WtdMegaritisXGBoost.OptimizationPresets.sustain_weartime` for separate
Optuna tuning within each outer fold. Both detector presets include human-only
three-fold ranking and independently sample 40% of human days plus five seed-42 simulated non-wear
training days. The script overrides this composition with its editable settings. Inner CV
holds out human participants with GroupKFold. Inner validation and candidate
ranking contain human days only. The optimizer samples 40% of each inner human
training set. `SIMULATED_NON_WEAR_DAY_COUNT=None` keeps all provided simulated non-wear training days unchanged; an
integer independently samples that many seeded simulated non-wear days from the provided
outer training half in the training dataset transform. The best candidate is
refitted on the full outer training fold, including its complete simulated non-wear training
half. Recording-level float32 features use TPCP's hybrid cache.

CNN trains directly in each outer fold. Its optional
`WtdMegaritisCNN.OptimizationPresets.sustain_weartime` provides a modest learning
rate/dropout/batch-size search for callers that want tuning; the CNN script does
not enable it. Windows are prepared lazily without
feature caching. GPU folds run sequentially.

Each run exports only `fold_results.csv`, `daily_results.csv`, and a final
trained model. Fold optimizers are not retained. After LOSO, the optimizer runs
once on the union of all selected human days and both simulated non-wear halves. This final
artifact fit happens after scoring and does not change the outer evaluation.
For XGBoost, this includes a new search with human-only inner validation followed by refitting
the best candidate on that complete training set. Its classifier and feature
order are saved as `model.pkl` and `feature_order.pkl`. CNN saves `model.keras`.
These final models are separate from the models used for held-out scoring.

Load CNN `.keras` artifacts in a fresh process with
`mobgap.weartime.load_keras_weartime_model(path)` to register the optional
model-side standardization layer before deserialization.

The signal-only revalidation uses the same seed-42 split of whole simulated
non-wear source recordings. Its result page keeps human and simulated non-wear
summaries separate. For non-wear, sample counts are pooled per recording in each
fold and false-wear minutes are averaged per evaluated day (not normalized to
24 hours). Fold-model results are averaged for each recording before equal
recording weighting and 95% t intervals. Repeated predictions do not add
independent observations to those intervals.
