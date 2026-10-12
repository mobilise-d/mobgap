# Wear-time training and evaluation

## Evaluation

The shared scripts are:

- `revalidation/weartime/_02_wtd_result_generation_no_exc.py`: generate signal,
  XGBoost and CNN results with the same outer participant LOSO.
- `revalidation/weartime/_01_wtd_analysis_no_exc.py`: compare their human and
  simulated non-wear performance separately.

All human days of one participant are held out together. Simulated non-wear
source recordings use one seed-42 50/50 split of complete original recordings;
all days of a recording stay in the same half. The same training and test
recordings repeat across outer fold models. The signal detector uses
`DummyOptimize`; CNN and XGBoost use `WearTimeOptunaSearch` with the trainers'
`OptimizationPresets.sustain_weartime`.

The inner presets rank three human participant folds. Each inner training set
independently samples 40% of human days and five seed-42 simulated non-wear days
from the supplied outer training pool. Simulated days never enter inner
validation. The best candidate is refitted on the complete outer training fold.
Every reference frame has a mandatory unordered categorical `label` column
(categories `["wear", "uncertain"]`), even when empty or all wear. Uncertain
means wear versus non-wear could not be determined; the unlabeled complement
is known non-wear. Trainers consume dataset datapoints and lazily access sensor data and labeled reference regions.

Partially uncertain human days remain in training: entire windows intersecting
uncertain intervals are excluded, retaining known windows on the original
sample grid. Held-out scoring masks uncertain samples as before.
Defaults use 20 Optuna trials; the CNN trainer fits for 60 epochs. Edit script
configuration before running. A full evaluation is expensive, especially
without a GPU.

Human metrics retain separate day, participant and fold weighting. Simulated
metrics pool sample counts per recording/fold, average false-wear minutes per
evaluated day (not normalized to 24 hours), then average fold models per
recording. Equal recording means and 95% t intervals use those unique recording
values; repeated predictions do not add independent CI observations. Only
current result files containing all configured algorithms and both populations
are supported.

## Production training

`train.py` tunes and refits XGBoost and CNN on the complete selected dataset.
There is no outer CV or held-out metric export. Inner search uses the same
SUSTAIN presets; final refitting uses all provided human and simulated days.
Edit its optimizer dictionary to choose algorithms and override model/search
settings. Final artifacts are exported under `.cache/weartime_models`:

- XGBoost: `WtdMegaritisXGBoost/xgboost_sustain_weartime_model.pkl` and
  `xgboost_sustain_weartime_model_config.json`. Its untrained selected classifier is saved as
  `xgboost_sustain_weartime_classifier_template.pkl`; fitting settings are in
  `xgboost_sustain_weartime_training_params.json`
- CNN: `WtdMegaritisCNN/cnn_sustain_weartime_model.keras` and
  `cnn_sustain_weartime_model_config.json`

Each model directory also contains its `*_best_params.json` with the selected
Optuna hyperparameters, using pipeline parameter names such as
`algo__clf__max_depth`. These are the searched settings, not a complete dump of
fixed model configuration.

To package a production run, copy these exported files into
`src/mobgap/weartime/production_models/`, retaining their names. The fixed
`WtdMegaritisXGBoost.PredefinedParameters.sustain_weartime` and
`WtdMegaritisCNN.PredefinedParameters.sustain_weartime` presets load
fitted models with their native sampling rates, window durations and ordered
inputs. The trainers' `PredefinedParameters.sustain_weartime` presets restore
fitting settings without loading weights. XGBoost's detector preset
`PredefinedParameters.untrained_sustain_weartime` restores the selected classifier
structure and inputs; `baseline_sustain_weartime` creates the canonical fresh baseline
for tuning before production artifacts exist. They require the freshly
trained artifacts to be packaged. Detection otherwise requires an explicitly
trained classifier/model, for example from
`Optimize(WtdMegaritisXGBoostTrainer()).optimize(dataset).optimized_pipeline_.algo`.
XGBoost classifier and input/feature settings belong to the supplied untrained detector. Its trainer owns
training overlap, recording processes and dataset conversion. Training overlap is independent of inference overlap.
Model inputs must use the native training rate; resample explicitly if needed.

Install MobGap with its `weartime` extra (TensorFlow requires Python below 3.13).
Set `DATASET_PATH` or `MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH` to the SUSTAIN wear-time
folder. `MOBGAP_CACHE_DIR_PATH` configures dataset/feature caches; evaluation
also requires `MOBGAP_VALIDATION_DATA_PATH`. The raw dataset is not distributed
with MobGap. Both workflows use local calendar days with at least eight hours
of recorded data.

XGBoost extracts float32 features per supplied recording/day using TPCP hybrid
caching. CNN prepares raw windows lazily without feature caching; per-window
standardization always runs inside its Keras model. Load exported CNN
artifacts in a fresh process with
`mobgap.weartime.load_keras_weartime_model(path)` to register its standardization
layer before deserialization.

The lightweight XGBoost feature list was selected from SHAP results pooled
across the historical SUSTAIN LOSO folds. This revalidation evaluates that
frozen method; its feature selection is not independent of this cohort.
