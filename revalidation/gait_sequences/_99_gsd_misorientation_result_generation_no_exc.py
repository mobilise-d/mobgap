"""
.. _gsd_misorientation_val_gen:

Revalidation of gait sequence detection under simulated mounting errors
========================================================================

.. note:: This is the code to create the results! If you are interested in
    viewing the results, please check the
    :ref:`results report <gsd_misorientation_val_results>`.

This script reproduces gait sequence detection validation results on the TVS
dataset while simulating every supported rough lower-back sensor mounting
orientation.

The TVS dataset does not provide recordings with known sensor mounting errors.
We therefore load each recording normally, convert the lower-back IMU signal to
body frame, apply the same rough rotations that are used in the reorientation
validation, and run the GSD algorithms on the rotated body-frame data.

The standard GSD validation script remains unchanged. This page writes its
results to a separate ``gsd_misorientation`` result folder.

.. warning::
    Before you modify and re-run this script, read through our guide on
    :ref:`revalidation`.
    In case you are planning to update official results, contact one of the
    core maintainers. They can assist with the process.

"""

# %%
# Setting up the algorithms
# -------------------------
# We only include executable Python algorithms here. The original MATLAB
# comparison algorithms used by the standard GSD validation are precomputed for
# the original TVS recordings and can therefore not be rerun on simulated
# mounting orientations.
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import Memory, Parallel, delayed
from mobgap import PROJECT_ROOT
from mobgap.data import TVSFreeLivingDataset, TVSLabDataset
from mobgap.gait_sequences import GsdAdaptiveIonescu, GsdIluz, GsdIonescu
from mobgap.gait_sequences.evaluation import gsd_score
from mobgap.gait_sequences.pipeline import GsdEmulationPipeline
from mobgap.re_orientation.evaluation import MisorientedDataset
from mobgap.utils.evaluation import Evaluation, save_evaluation_results
from mobgap.utils.misc import get_env_var

pipelines = {
    "GsdIluz": GsdEmulationPipeline(GsdIluz(), convert_to_body_frame=False),
    "GsdIluz_orig_peak": GsdEmulationPipeline(
        GsdIluz(**GsdIluz.PredefinedParameters.original),
        convert_to_body_frame=False,
    ),
    "GsdIonescu": GsdEmulationPipeline(
        GsdIonescu(), convert_to_body_frame=False
    ),
    "GsdAdaptiveIonescu": GsdEmulationPipeline(
        GsdAdaptiveIonescu(),
        convert_to_body_frame=False,
    ),
}

# %%
# Setting up the dataset
# ----------------------
# ``MisorientedDataset`` expands each TVS datapoint by an additional
# ``orientation`` index column. With ``output_frame="body"``, every algorithm
# receives body-frame data after the simulated rough mounting rotation.
cache_dir = Path(get_env_var("MOBGAP_CACHE_DIR_PATH", PROJECT_ROOT / ".cache"))

datasets_free_living = MisorientedDataset(
    TVSFreeLivingDataset(
        get_env_var("MOBGAP_TVS_DATASET_PATH"),
        reference_system="INDIP",
        memory=Memory(cache_dir),
        missing_reference_error_type="skip",
    ),
    output_frame="body",
)
datasets_laboratory = MisorientedDataset(
    TVSLabDataset(
        get_env_var("MOBGAP_TVS_DATASET_PATH"),
        reference_system="INDIP",
        memory=Memory(cache_dir),
        missing_reference_error_type="skip",
    ),
    output_frame="body",
)

# %%
# Running the evaluation
# ----------------------
# Each algorithm is evaluated on the orientation-expanded dataset. The
# per-recording/per-trial ``single_results.csv`` files are the primary output of
# this validation because their index includes the simulated orientation.
n_jobs = int(get_env_var("MOBGAP_N_JOBS", 3))
results_base_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH"))
    / "results/gsd_misorientation"
)


def run_evaluation(
    name: str,
    pipeline: GsdEmulationPipeline,
    ds: MisorientedDataset,
) -> tuple[str, Evaluation[GsdEmulationPipeline]]:
    eval_pipe = Evaluation(ds, scoring=gsd_score).run(pipeline)
    return name, eval_pipe


def eval_debug_plot(
    results: dict[str, Evaluation[GsdEmulationPipeline]],
) -> None:
    results_df = (
        pd.concat({k: v.get_single_results_as_df() for k, v in results.items()})
        .reset_index()
        .rename(columns={"level_0": "algo_name"})
    )

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    sns.boxplot(
        data=results_df,
        x="algo_name",
        y="f1_score",
        hue="orientation",
        showmeans=True,
        ax=ax,
    )
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("F1 score")
    ax.tick_params(axis="x", rotation=20)
    plt.show()


# %%
# Free-Living
# ~~~~~~~~~~~
with Parallel(n_jobs=n_jobs) as parallel:
    results_free_living: dict[str, Evaluation[GsdEmulationPipeline]] = dict(
        parallel(
            delayed(run_evaluation)(name, pipeline, datasets_free_living)
            for name, pipeline in pipelines.items()
        )
    )

# %%
# We create a quick plot for debugging.
eval_debug_plot(results_free_living)

# %%
# Then we save the results to disk.
for k, v in results_free_living.items():
    save_evaluation_results(
        k,
        v,
        condition="free_living",
        base_path=results_base_path,
        raw_results=["detected"],
    )

# %%
# Laboratory
# ~~~~~~~~~~
with Parallel(n_jobs=n_jobs) as parallel:
    results_laboratory: dict[str, Evaluation[GsdEmulationPipeline]] = dict(
        parallel(
            delayed(run_evaluation)(name, pipeline, datasets_laboratory)
            for name, pipeline in pipelines.items()
        )
    )

# %%
# We create a quick plot for debugging.
eval_debug_plot(results_laboratory)

# %%
# Then we save the results to disk.
for k, v in results_laboratory.items():
    save_evaluation_results(
        k,
        v,
        condition="laboratory",
        base_path=results_base_path,
        raw_results=["detected"],
    )
