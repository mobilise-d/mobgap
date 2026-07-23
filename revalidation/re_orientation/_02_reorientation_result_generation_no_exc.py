"""
.. _reorientation_val_gen:

Revalidation of the reorientation algorithm
===========================================

.. note:: This is the code to create the results! If you are interested in viewing the
    results, please check the results report.

This script reproduces the validation results for the reorientation algorithm on the
TVS dataset.

The TVS dataset does not provide a reference for intentionally misoriented lower-back
sensor recordings. We therefore use the reference walking bouts, simulate every
supported rough mounting orientation with :func:`flip_dataset`, and evaluate the
algorithm response as a multiclass classification task.

Performance metrics are calculated on a per-trial/per-recording basis and aggregated
over the whole dataset. The raw orientation predictions and all performance metrics
are saved to disk.

.. warning::
    Before you modify and re-run this script, read through our guide on
    :ref:`revalidation`.
    In case you are planning to update the official results, contact one of the core
    maintainers. They can assist with the process.

"""

# %%
# Setting up the algorithms
# -------------------------
# We use the :class:`~mobgap.re_orientation.pipeline.ReorientationEmulationPipeline`
# to run the algorithm on simulated misorientations.
#
# .. note:: Set up your environment variables to point to the correct paths.
#    The easiest way to do this is to create a `.env` file in the root of the
#    repository with the following content. You need the paths to the root folder of
#    the TVS dataset `MOBGAP_TVS_DATASET_PATH` and the path where revalidation results
#    should be stored `MOBGAP_VALIDATION_DATA_PATH`. The path to the cache directory
#    `MOBGAP_CACHE_DIR_PATH` is optional.
from mobgap.re_orientation import (
    ReorientationEmulationPipeline,
    ReorientationMethodDM,
)
from mobgap.utils.misc import get_env_var

pipelines = {
    "MethodDM__full": ReorientationEmulationPipeline(
        ReorientationMethodDM(correction_mode="full")
    ),
    "MethodDM__trust_gravity": ReorientationEmulationPipeline(
        ReorientationMethodDM(correction_mode="trust_gravity")
    ),
}

# %%
# Setting up the dataset
# ----------------------
# We run the comparison on the Lab and the Free-Living part of the TVS dataset.
# We use the reference walking bouts from the INDIP reference system and simulate
# sensor misorientations within each reference walking bout.
from pathlib import Path

from joblib import Memory
from mobgap import PROJECT_ROOT
from mobgap.data import TVSFreeLivingDataset, TVSLabDataset

cache_dir = Path(get_env_var("MOBGAP_CACHE_DIR_PATH", PROJECT_ROOT / ".cache"))

datasets_free_living = TVSFreeLivingDataset(
    get_env_var("MOBGAP_TVS_DATASET_PATH"),
    reference_system="INDIP",
    memory=Memory(cache_dir),
    missing_reference_error_type="skip",
)
datasets_laboratory = TVSLabDataset(
    get_env_var("MOBGAP_TVS_DATASET_PATH"),
    reference_system="INDIP",
    memory=Memory(cache_dir),
    missing_reference_error_type="skip",
)


# %%
# Running the evaluation
# ----------------------
# We multiprocess the evaluation on the level of algorithms using tpcp's
# context-aware joblib helpers. Each algorithm pipeline is run using its own
# instance of the
# :class:`~mobgap.utils.evaluation.Evaluation` class.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mobgap.re_orientation.evaluation import reorientation_score
from mobgap.utils.evaluation import Evaluation
from tpcp.parallel import Parallel

from revalidation._utils import create_evaluation_tasks

n_jobs = int(get_env_var("MOBGAP_N_JOBS", 3))
results_base_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH")) / "results/re_orientation"
)


def run_evaluation(name, pipeline, ds):
    eval_pipe = Evaluation(
        ds,
        scoring=reorientation_score,
    ).run(pipeline)
    return name, eval_pipe


# %%


def eval_debug_plot(
    results: dict[str, Evaluation[ReorientationEmulationPipeline]],
) -> None:
    results_df = (
        pd.concat({k: v.get_single_results_as_df() for k, v in results.items()})
        .reset_index()
        .rename(columns={"level_0": "algo_name"})
    )

    sns.boxplot(
        data=results_df,
        x="cohort",
        y="accuracy",
        hue="algo_name",
        showmeans=True,
    )
    plt.tight_layout()
    plt.show()


# %%
# Free-Living
# ~~~~~~~~~~~
# Let's start with the Free-Living part of the dataset.
with Parallel(n_jobs=n_jobs) as parallel:
    results_free_living: dict[
        str, Evaluation[ReorientationEmulationPipeline]
    ] = dict(
        parallel(
            create_evaluation_tasks(
                run_evaluation,
                pipelines,
                datasets_free_living,
                condition="free_living",
            )
        )
    )

# %%
# We create a quick plot for debugging.
eval_debug_plot(results_free_living)

# %%
# Then we save the results to disk.
from mobgap.utils.evaluation import save_evaluation_results

for k, v in results_free_living.items():
    save_evaluation_results(
        k,
        v,
        condition="free_living",
        base_path=results_base_path,
        raw_results=["predictions", "confusion_matrix"],
    )


# %%
# Laboratory
# ~~~~~~~~~~
# Now, we repeat the evaluation for the Laboratory part of the dataset.
with Parallel(n_jobs=n_jobs) as parallel:
    results_laboratory: dict[
        str, Evaluation[ReorientationEmulationPipeline]
    ] = dict(
        parallel(
            create_evaluation_tasks(
                run_evaluation,
                pipelines,
                datasets_laboratory,
                condition="laboratory",
            )
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
        raw_results=["predictions", "confusion_matrix"],
    )
