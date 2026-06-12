"""Full-pipeline validation under simulated mounting errors.

.. _pipeline_val_mounting_error_gen:

Full-pipeline validation under simulated mounting errors
=======================================================

.. note::
    This is the code to create the results.

This script runs the Mobilise-D full pipeline on the free-living TVS dataset
while simulating each supported rough lower-back sensor mounting orientation.
The underlying TVS dataset is loaded normally and then wrapped with
``MisorientedDataset``. This keeps the original data-loading cache reusable and
applies the simulated full-recording rotation only when a datapoint is accessed.

The wrapped dataset returns the same frame as the wrapped TVS dataset. For TVS,
this means the pipeline receives sensor-frame data and performs its standard
sensor-to-body-frame conversion internally.

The comparison contains three variants:

* the current default Mobilise-D pipeline,
* the current default Mobilise-D pipeline with reorientation correction enabled,
* the same reorientation-enabled pipeline, but with
  :class:`~mobgap.gait_sequences.GsdIluzAdaptiveGravity` in the regular-walking
  sub-pipeline.

Only the free-living condition is evaluated because this is the intended use
case for unknown mounting orientations.

.. warning::
    Before you modify and re-run this script, read through our guide on
    :ref:`revalidation`.
    In case you are planning to update official results, contact one of the
    core maintainers. They can assist with the process.

"""

# %%
# Setting Up The Pipelines
# ------------------------
# We compare the current default full pipeline against two variants that enable
# per-gait-sequence reorientation correction. The adaptive-GSD variant only
# changes the regular-walking sub-pipeline; the impaired sub-pipeline stays on
# the current ``GsdIonescu`` setup.
from pathlib import Path

from joblib import Memory, parallel_backend
from mobgap import PROJECT_ROOT
from mobgap.data import TVSFreeLivingDataset
from mobgap.gait_sequences import GsdIluzAdaptiveGravity
from mobgap.pipeline import (
    MobilisedPipelineHealthy,
    MobilisedPipelineImpaired,
    MobilisedPipelineUniversal,
)
from mobgap.pipeline.base import BaseMobilisedPipeline
from mobgap.pipeline.evaluation import pipeline_score
from mobgap.re_orientation import ReorientationMethodDM
from mobgap.utils.evaluation import Evaluation, save_evaluation_results
from mobgap.utils.misc import get_env_var

from revalidation.full_pipeline._orientation_dataset import MisorientedDataset

pipelines = {
    "Official_MobiliseD_Pipeline": MobilisedPipelineUniversal(),
    "Official_MobiliseD_Pipeline__reorientation": MobilisedPipelineUniversal(
        pipelines=[
            (
                "healthy",
                MobilisedPipelineHealthy(
                    reorientation_correction=ReorientationMethodDM(
                        correction_mode="full"
                    ),
                ),
            ),
            (
                "impaired",
                MobilisedPipelineImpaired(
                    reorientation_correction=ReorientationMethodDM(
                        correction_mode="full"
                    ),
                ),
            ),
        ]
    ),
    "Official_MobiliseD_Pipeline__reorientation_gsd_iluz_adaptive": (
        MobilisedPipelineUniversal(
            pipelines=[
                (
                    "healthy",
                    MobilisedPipelineHealthy(
                        gait_sequence_detection=GsdIluzAdaptiveGravity(
                            expected_pa_axis="pa"
                        ),
                        reorientation_correction=ReorientationMethodDM(
                            correction_mode="full"
                        ),
                    ),
                ),
                (
                    "impaired",
                    MobilisedPipelineImpaired(
                        reorientation_correction=ReorientationMethodDM(
                            correction_mode="full"
                        ),
                    ),
                ),
            ]
        )
    ),
}

# %%
# Setting Up The Dataset
# ----------------------
# We only evaluate the free-living TVS recordings. ``MisorientedDataset``
# expands each recording by an additional ``orientation`` index column and
# returns one rotated full recording for each supported simulated mounting
# orientation.
cache_dir = Path(get_env_var("MOBGAP_CACHE_DIR_PATH", PROJECT_ROOT / ".cache"))

datasets_free_living = MisorientedDataset(
    TVSFreeLivingDataset(
        get_env_var("MOBGAP_TVS_DATASET_PATH"),
        reference_system="INDIP",
        memory=Memory(cache_dir),
        missing_reference_error_type="skip",
    )
)

# %%
# Running The Evaluation
# ----------------------
# We run the pipeline variants one after another and use multiprocessing within
# each evaluation. This parallelizes over datapoints, which gives better CPU
# utilization than splitting only across the three pipeline variants.

n_jobs = int(get_env_var("MOBGAP_N_JOBS", 3))
results_base_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH"))
    / "results/full_pipeline_mounting_error"
)


def run_evaluation(
    name: str,
    pipeline: BaseMobilisedPipeline,
    ds: MisorientedDataset,
) -> tuple[str, Evaluation[BaseMobilisedPipeline]]:
    scoring = pipeline_score.clone().set_params(n_jobs=n_jobs, verbose=10)
    # tpcp.validate resets explicit Scorer multiprocessing params to its own
    # defaults. The backend context keeps the scorer's internal Parallel call
    # on the intended process pool.
    with parallel_backend("loky", n_jobs=n_jobs):
        eval_pipe = Evaluation(
            ds,
            scoring=scoring,
        ).run(pipeline)
    return name, eval_pipe


# %%
# Free-Living
# ~~~~~~~~~~~
# Results are written after each pipeline variant finishes so that completed
# variants remain available if a later, slower variant is interrupted.
for name, pipeline in pipelines.items():
    _, result = run_evaluation(name, pipeline, datasets_free_living)
    save_evaluation_results(
        name,
        result,
        condition="free_living",
        base_path=results_base_path,
        raw_results=["matched_errors"],
    )
