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
`DummyOptimize`; CNN and XGBoost use `WearTimeOptunaSearch` with their
`OptimizationPresets.sustain_weartime`.

The inner presets rank three human participant folds. Each inner training set
independently samples 40% of human days and five seed-42 simulated non-wear days
from the supplied outer training pool. Simulated days never enter inner
validation. The best candidate is refitted on the complete outer training fold.
Every reference frame has a mandatory unordered categorical `label` column
(categories `["wear", "uncertain"]`), even when empty or all wear. Uncertain
means wear versus non-wear could not be determined; the unlabeled complement
is known non-wear. Training consumes `(data, labeled_reference)` pairs.

Partially uncertain human days remain in training: entire windows intersecting
uncertain intervals are excluded, retaining known windows on the original
sample grid. Held-out scoring masks uncertain samples as before.
Defaults use 20 Optuna trials; the CNN model trains for 60 epochs. Edit script
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

- XGBoost: `WtdMegaritisXGBoost/model.pkl` and `feature_order.pkl`
- CNN: `WtdMegaritisCNN/model.keras`

Install MobGap with its `weartime` extra (TensorFlow requires Python below 3.13).
Set `DATASET_PATH` or `MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH` to the SUSTAIN wear-time
folder. `MOBGAP_CACHE_DIR_PATH` configures dataset/feature caches; evaluation
also requires `MOBGAP_VALIDATION_DATA_PATH`. The raw dataset is not distributed
with MobGap. Both workflows use local calendar days with at least eight hours
of recorded data.

XGBoost extracts float32 features per supplied recording/day using TPCP hybrid
caching. CNN prepares windows lazily without feature caching. Load exported CNN
artifacts in a fresh process with
`mobgap.weartime.load_keras_weartime_model(path)` to register its standardization
layer before deserialization.
