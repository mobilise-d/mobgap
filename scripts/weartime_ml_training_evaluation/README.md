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

`train.py` tunes and refits XGBoost, CNN and CNN-LSTM on the complete selected dataset.
There is no outer CV or held-out metric export. Inner search uses the SUSTAIN presets;
final refitting uses all supplied human and simulated days. Edit the optimizer dictionary
before running. CNN uses 60 epochs; CNN-LSTM uses 57 epochs and Adam gradient clipping at 1.

The supplied untrained detector owns its classifier/network type, structure and input
configuration. `MegaritisCnnWeartimeModel` and `MegaritisCnnLstmWeartimeModel` build their
own graphs. The CNN trainer owns Adam settings, epochs, batch/shuffle settings, seed,
training overlap and dataset conversion. XGBoost's trainer owns training overlap,
recording processes and dataset conversion; native estimator parameters stay on `algo.clf`.
Inference overlap, thresholds and batching remain on the detector/model. An untrained input
configuration may have no sampling rate; fitting fills the dataset's actual rate. Fitted
inference requires that native rate, so resample explicitly if needed.

Final artifacts under `.cache/weartime_models` use these independent folders/prefixes:

| Configuration | Folder | Artifact prefix |
| --- | --- | --- |
| XGBoost | `WtdMegaritisXGBoost` | `xgboost_sustain_weartime` |
| CNN | `WtdMegaritisCNN` | `cnn_sustain_weartime` |
| CNN-LSTM | `WtdMegaritisCNNLSTM` | `cnn_lstm_sustain_weartime` |

Each exports `*_model_config.json` (native rate and ordered inputs),
`*_best_params.json` (winning searched values, including nested model parameters) and
`*_training_params.json` (complete fitting recipe, excluding the algorithm).
CNN variants export `*_model.keras` and `*_model_params.json` (complete window-model
constructor settings, excluding fitted weights and input configuration).
XGBoost exports `*_model.pkl` and `*_classifier_template.pkl`; the latter contains
selected estimator constructor parameters with fitted state removed by sklearn cloning.

To package a production run, copy its exported files into
`src/mobgap/weartime/production_models/`, retaining their names. Fitted preset loaders
require those freshly trained artifacts; none are generated automatically on import.

- `WtdMegaritisXGBoost.PredefinedParameters.sustain_weartime` loads fitted weights and inputs.
  `untrained_sustain_weartime` loads the selected untrained template and inputs;
  `baseline_sustain_weartime` creates the canonical fresh baseline before artifacts exist.
- `WtdMegaritisCNN.PredefinedParameters.sustain_weartime_cnn` and `sustain_weartime_cnn_lstm`
  load their respective fitted models. `untrained_sustain_weartime_cnn` and
  `untrained_sustain_weartime_cnn_lstm` load the respective model definitions without weights.
- Trainer presets load only fitting recipes: XGBoost's `sustain_weartime`, and CNN's
  `sustain_weartime_cnn` / `sustain_weartime_cnn_lstm`. Supply the matching untrained detector
  when constructing a trainer from a recipe. Search spaces and inner CV remain in
  `OptimizationPresets.sustain_weartime`.

For example, retrain the selected CNN-LSTM definition through TPCP:

```python
algo = WtdMegaritisCNN(**WtdMegaritisCNN.PredefinedParameters.untrained_sustain_weartime_cnn_lstm)
trainer = WtdMegaritisCNNTrainer(
    algo,
    **WtdMegaritisCNNTrainer.PredefinedParameters.sustain_weartime_cnn_lstm,
)
fitted_detector = Optimize(trainer).optimize(dataset).optimized_pipeline_.algo
```

Install MobGap with its `weartime` extra (TensorFlow requires Python below 3.13).
Set `DATASET_PATH` or `MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH` to the SUSTAIN wear-time
folder. `MOBGAP_CACHE_DIR_PATH` configures dataset/feature caches; evaluation
also requires `MOBGAP_VALIDATION_DATA_PATH`. The raw dataset is not distributed
with MobGap. Both workflows use local calendar days with at least eight hours
of recorded data.

XGBoost extracts float32 features per supplied recording/day using TPCP hybrid
caching. CNN prepares raw windows lazily without feature caching; per-window
standardization runs inside the built-in Megaritis CNN and CNN-LSTM graphs.
Supplied native Keras graphs receive raw windows during fitting and prediction
and must contain their own required preprocessing. Load exported CNN
artifacts in a fresh process with
`mobgap.weartime.load_keras_weartime_model(path)` to register its standardization
layer before deserialization.

The lightweight XGBoost feature list was selected from SHAP results pooled
across the historical SUSTAIN LOSO folds. This revalidation evaluates that
frozen method; its feature selection is not independent of this cohort.
