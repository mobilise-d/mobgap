"""
.. _pipeline_val_misorientation_gen:

Full-pipeline validation under simulated mounting errors
========================================================

.. note::
    This is the code to create the results.

This script runs the Mobilise-D full pipeline on the free-living TVS dataset
while simulating each supported rough lower-back sensor mounting orientation.
The underlying TVS dataset is loaded normally and then wrapped with
``MisorientedDataset``. This keeps the original data-loading cache reusable and
applies the simulated full-recording rotation only when a datapoint is accessed.

The comparison contains two variants:

* the default :class:`~mobgap.pipeline.MobilisedPipelineUniversal` without
  per-GS reorientation and with the usual cohort-specific GSD choices,
* a reorientation-enabled variant where both cohort-specific sub-pipelines use
  :class:`~mobgap.gait_sequences.GsdIonescu` and
  :class:`~mobgap.re_orientation.ReorientationMethodDM`.

The default regular-walking GSD, :class:`~mobgap.gait_sequences.GsdIluz`, is
therefore present in the default pipeline only. It is not evaluated with
reorientation enabled because ``GsdIluz`` requires body-frame input, while
per-GS reorientation requires the GSD step to work on the unknown sensor frame
before correction.

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
# The default pipeline keeps the usual cohort-specific GSDs. The
# reorientation-enabled variant explicitly uses ``GsdIonescu`` in both
# sub-pipelines because ``GsdIluz`` is orientation-dependent and cannot be used
# when reorientation happens after GSD.
from pathlib import Path

from joblib import Memory
from mobgap import PROJECT_ROOT
from mobgap.data import TVSFreeLivingDataset
from mobgap.gait_sequences import GsdIonescu
from mobgap.pipeline import (
    MobilisedPipelineHealthy,
    MobilisedPipelineImpaired,
    MobilisedPipelineUniversal,
)
from mobgap.pipeline.base import BaseMobilisedPipeline
from mobgap.pipeline.evaluation import pipeline_score
from mobgap.re_orientation import ReorientationMethodDM
from mobgap.re_orientation.evaluation import MisorientedDataset
from mobgap.utils.evaluation import Evaluation, save_evaluation_results
from mobgap.utils.misc import get_env_var

pipelines = {
    "Official_MobiliseD_Pipeline": MobilisedPipelineUniversal(),
    (
        "Official_MobiliseD_Pipeline__gsd_ionescu_reorientation"
    ): MobilisedPipelineUniversal(
        pipelines=[
            (
                "healthy",
                MobilisedPipelineHealthy(
                    gait_sequence_detection=GsdIonescu(),
                    per_gs_reorientation=ReorientationMethodDM(),
                ),
            ),
            (
                "impaired",
                MobilisedPipelineImpaired(
                    gait_sequence_detection=GsdIonescu(),
                    per_gs_reorientation=ReorientationMethodDM(),
                ),
            ),
        ]
    ),
}

# %%
# Setting Up The Dataset
# ----------------------
# ``MisorientedDataset`` expands each TVS datapoint by an additional
# ``orientation`` index column. It returns sensor-frame TVS data with the
# simulated mounting error applied, so the full pipeline still receives data in
# the same frame it expects from the underlying TVS dataset.
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
# We run the pipeline variants one after another and use datapoint-level
# multiprocessing inside ``tpcp.validate``. Results are written after each
# variant finishes so completed results remain available if a later run is
# interrupted.
n_jobs = int(get_env_var("MOBGAP_N_JOBS", 3))
results_base_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH"))
    / "results/full_pipeline_misorientation"
)


def run_evaluation(
    name: str,
    pipeline: BaseMobilisedPipeline,
    ds: MisorientedDataset,
) -> tuple[str, Evaluation[BaseMobilisedPipeline]]:
    eval_pipe = Evaluation(
        ds,
        scoring=pipeline_score,
        validate_paras={"n_jobs": n_jobs, "verbose": 10},
    ).run(pipeline)
    return name, eval_pipe


# %%
# Free-Living
# ~~~~~~~~~~~
for name, pipeline in pipelines.items():
    _, result = run_evaluation(name, pipeline, datasets_free_living)
    save_evaluation_results(
        name,
        result,
        condition="free_living",
        base_path=results_base_path,
        raw_results=["matched_errors"],
    )
