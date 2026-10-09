"""
.. _wtd_val_gen_no_exc:

Revalidation of the wear-time detection algorithms
==================================================

This script evaluates signal, XGBoost and CNN detectors on SUSTAIN human
recordings
with participant LOSO. Each held-out participant contributes daily datapoints
with at least eight hours of recorded data. Simulated non-wear source recordings
use one fixed seed-42 50/50 train/test split, repeated in every human fold.
All evaluated days from one simulated recording remain in the same half.
The signal detector uses ``DummyOptimize``. XGBoost and CNN use the SUSTAIN
Optuna presets to tune on human-only inner validation and refit each outer
training fold. The default CNN fits 60 epochs and each search runs 20 trials.

Per-day and per-fold metrics and raw interval matches are saved locally. The
analysis summarizes labeled-sample confusion counts with day, participant and
fold weighting, and averages daily errors for duration summaries. Participant
010's uncertain ground truth is excluded from scoring; the detector still
receives its full signal.
Undefined daily metrics are NaN.

.. warning::
    Read :ref:`revalidation` before changing this script. These results are
    unpublished; keep the ``_no_exc`` suffix until they are intended for the
    public validation-result release.

"""

# %%
# Configure the signal detector and the daily dataset
# --------------------------------------------------
from pathlib import Path

from joblib import Memory
from mobgap import PROJECT_ROOT
from mobgap.data import SustainWearTimeDataset
from mobgap.utils.evaluation import EvaluationCV, save_evaluation_results
from mobgap.utils.misc import get_env_var
from mobgap.weartime import (
    MegaritisCnnWeartimeModel,
    WtdMegaritisCNN,
    WtdMegaritisSignal,
    WtdMegaritisXGBoost,
)
from mobgap.weartime.evaluation import wtd_score
from mobgap.weartime.optimization import WearTimeOptunaSearch
from mobgap.weartime.pipeline import WtdEmulationPipeline
from sklearn.model_selection import LeaveOneGroupOut
from tpcp.optimize import DummyOptimize
from tpcp.validate import (
    CombinedSplitter,
    DatasetSplitter,
    NoSplit,
    SubsetSplitter,
)

cache_dir = Path(
    get_env_var("MOBGAP_CACHE_DIR_PATH", PROJECT_ROOT / ".cache")
).expanduser()
results_base_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH"))
    / "results/weartime_loso_no_exc_min8h"
)
condition_name = "sustain_weartime"
n_jobs = int(get_env_var("MOBGAP_N_JOBS", 1))
SEED = 42

dataset_sustain_weartime = SustainWearTimeDataset(
    Path(get_env_var("MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH")).expanduser(),
    tz="Europe/London",
    additional_sensors_enabled=(),
    memory=Memory(cache_dir),
)
# The frozen lightweight XGBoost feature list was selected using this cohort;
# revalidation does not provide an independent feature-selection assessment.
optimizers = {
    "WtdMegaritisSignal": DummyOptimize(
        WtdEmulationPipeline(WtdMegaritisSignal()),
        ignore_potential_user_error_warning=True,
    ),
    "WtdMegaritisXGBoost": WearTimeOptunaSearch(
        WtdEmulationPipeline(
            WtdMegaritisXGBoost(
                **WtdMegaritisXGBoost.PredefinedParameters.untrained_lightweight,
                memory=Memory(
                    cache_dir / "xgboost_features", compress=3, verbose=0
                ),
            )
        ),
        **WtdMegaritisXGBoost.OptimizationPresets.sustain_weartime,
    ),
    "WtdMegaritisCNN": WearTimeOptunaSearch(
        WtdEmulationPipeline(
            WtdMegaritisCNN(
                model=MegaritisCnnWeartimeModel(standardize_in_model=True)
            )
        ),
        **WtdMegaritisCNN.OptimizationPresets.sustain_weartime,
    ),
}

# %%
# Hold out every day of one human participant per fold
# ---------------------------------------------------
# The dataset contains two parts: human recordings and simulated non-wear
# recordings.
# The human recordings are the most important part of our evaluation.
# We use a leave one participant out (LOSO) cross-validation scheme to
# estimate the performance on independent human participants.
#
# The dataset further contains simulated non-wear recordings.
# As this part of the data contans "unrealistic" distributions of wear vs.
# non-wear, we would not expect during real deployments,
# we try to keep this part seperated during the evaluation.
# However, this data might still be usefull for the training of the models to
# provide them with examples of non-wear "movements".
#
# For this reason we use a very unusal cross-validation approach.
# We split the actual human recordings using a LOSO scheme.
# Then we add a fixed 50% of the simulated non-wear recordings to eachs fold
# train data, so that the models can learn from this data.
#
# To allow evaluation on the non-wear recordings, we add the remaining 50% of
# the simulated non-wear recordings to each fold test data.
# This means each test set contains one unique human participant and the same
# 50% of the simulated non-wear recordings.
# We will later sperate the two parts of the test data again before
# calculating the metrics.
# But this approach allows us to get all results in a single run of the
# evaluation, without complex managment of intermediate model artifacts or
# similar.
splitter = CombinedSplitter(
    parts=[
        (
            "human",
            SubsetSplitter(
                lambda days: days.get_subset(recording_type="human_movement"),
                DatasetSplitter(LeaveOneGroupOut(), groupby="participant_id"),
            ),
        ),
        (
            "simulated_non_wear",
            SubsetSplitter(
                lambda days: days.get_subset(
                    recording_type="simulated_movements"
                ),
                NoSplit(
                    None,
                    train=lambda days: days.get_subset(
                        recording_id=days.index["recording_id"]
                        .drop_duplicates()
                        .sort_values()
                        .sample(frac=1, random_state=SEED)
                        .iloc[: days.index["recording_id"].nunique() // 2]
                        .tolist()
                    ),
                    test=lambda days: days.get_subset(
                        recording_id=days.index["recording_id"]
                        .drop_duplicates()
                        .sort_values()
                        .sample(frac=1, random_state=SEED)
                        .iloc[days.index["recording_id"].nunique() // 2 :]
                        .tolist()
                    ),
                ),
            ),
        ),
    ]
)

# %%
# Evaluate held-out days and save the standard result tables
# ---------------------------------------------------------
results_sustain_weartime = {
    name: EvaluationCV(
        dataset_sustain_weartime,
        scoring=wtd_score,
        cv_iterator=splitter,
        cv_params={"n_jobs": n_jobs, "return_train_score": False},
    ).run(optimizer)
    for name, optimizer in optimizers.items()
}

for name, result in results_sustain_weartime.items():
    save_evaluation_results(
        name,
        result,
        condition=condition_name,
        base_path=results_base_path,
        raw_results=[
            "matches",
            "detected",
            "detected_scored",
            "reference",
            "reference_scored",
            "reference_waking",
        ],
        include_non_stable_results=True,
    )
