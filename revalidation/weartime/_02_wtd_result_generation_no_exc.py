"""
.. _wtd_val_gen_no_exc:

Revalidation of the wear-time detection algorithms
==================================================

This script evaluates the signal-based detector on SUSTAIN human recordings
with participant LOSO. Each held-out participant contributes daily datapoints
with at least eight hours of recorded data. Simulated non-wear recordings
are excluded.
The signal detector has no fitted parameters, so ``DummyOptimize`` runs the same
configured detector in every fold. This establishes the CV workflow for future
trainable detectors.

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
from mobgap.weartime import WtdMegaritisSignal
from mobgap.weartime.evaluation import wtd_score
from mobgap.weartime.pipeline import WtdEmulationPipeline
from sklearn.model_selection import LeaveOneGroupOut
from tpcp.optimize import DummyOptimize
from tpcp.validate import CombinedSplitter, DatasetSplitter, SubsetSplitter

cache_dir = Path(get_env_var("MOBGAP_CACHE_DIR_PATH", PROJECT_ROOT / ".cache"))
results_base_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH"))
    / "results/weartime_loso_no_exc_min8h"
)
condition_name = "sustain_weartime"
n_jobs = int(get_env_var("MOBGAP_N_JOBS", 3))

dataset_sustain_weartime = SustainWearTimeDataset(
    get_env_var("MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH"),
    tz="Europe/London",
    additional_sensors_enabled=(),
    memory=Memory(cache_dir),
)
optimizers = {
    "WtdMegaritisSignal": DummyOptimize(
        WtdEmulationPipeline(WtdMegaritisSignal()),
        ignore_potential_user_error_warning=True,
    ),
}

# %%
# Hold out every day of one human participant per fold
# ---------------------------------------------------
# Selection belongs to the splitter. Simulated non-wear recordings enter neither
# train nor test sets.
splitter = CombinedSplitter(
    parts=[
        (
            "human",
            SubsetSplitter(
                lambda days: days.get_subset(recording_type="human_movement"),
                DatasetSplitter(LeaveOneGroupOut(), groupby="participant_id"),
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
