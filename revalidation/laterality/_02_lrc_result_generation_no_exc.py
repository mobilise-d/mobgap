"""
.. _lrc_val_gen:

Revalidation of the laterality classification algorithms
========================================================

.. note:: This is the code to create the results! If you are interested in viewing the results, please check the
    :ref:`results report <lrc_val_results>`.

This script reproduces the validation results on TVS dataset for the laterality detection algorithms.

Performance metrics are calculated on a per-trial/per-recording basis and aggregated (mean for most metrics)
over the whole dataset.
The raw detected initial contacts and all performance metrics are saved to disk.

.. warning:: Before you modify and re-run this script, read through our guide on :ref:`revalidation`.
   In case you are planning to update the official results (either after a code change, or because an algorithm was
   added), contact one of the core maintainers.
   They can assist with the process.

"""

# %%
# Setting up the algorithms
# -------------------------
# We use the :class:`~mobgap.initial_contacts.pipeline.IcdEmulationPipeline` to run the algorithms.
# We create an instance of this pipeline for each algorithm we want to evaluate and store them in a dictionary.
# The key is used to identify the algorithm in the results and used as folder name to store the results.
#
# .. note:: Set up your environment variables to point to the correct paths.
#    The easiest way to do this is to create a `.env` file in the root of the repository with the following content.
#    You need the paths to the root folder of the TVS dataset `MOBGAP_TVS_DATASET_PATH` and the path where revalidation
#    results should be stored `MOBGAP_VALIDATION_DATA_PATH`.
#    The path to the cache directory `MOBGAP_CACHE_DIR_PATH` is optional, when you don't want to store the memory cache
#    in the default location.
from mobgap.laterality import LrcMcCamley, LrcUllrich
from mobgap.laterality.pipeline import LrcEmulationPipeline
from mobgap.utils.misc import get_env_var

pipelines = {
    "McCamley": LrcEmulationPipeline(LrcMcCamley()),
    "UllrichOld__ms_all": LrcEmulationPipeline(
        LrcUllrich(**LrcUllrich.PredefinedParameters.msproject_all_old)
    ),
    "UllrichOld__ms_ms": LrcEmulationPipeline(
        LrcUllrich(**LrcUllrich.PredefinedParameters.msproject_ms_old)
    ),
    "UllrichNew__ms_all": LrcEmulationPipeline(
        LrcUllrich(**LrcUllrich.PredefinedParameters.msproject_all)
    ),
}

# %%
# Setting up the dataset
# ----------------------
# We run the comparison on the Lab and the Free-Living part of the TVS dataset.
# We use the :class:`~mobgap.data.TVSFreeLivingDataset` and the :class:`~mobgap.data.TVSLabDataset` to load the data.
# Note, that we use Memory caching to speed up the loading of the data.
# We also skip the recordings where the reference data is missing.
# In both cases, we compare against the INDIP reference system as done in the original validation as well.
#
# In the evaluation, each row of the dataset is treated as a separate recording.
# Results are calculated per recording.
# Aggregated results are calculated over the whole dataset, without considering the content of the individual
# recordings.
# Depending on how you want to interpret the results, you might not want to use the aggregated results, but rather
# perform custom aggregations over the provided "single_results".
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
# We multiprocess the evaluation on the level of algorithms using joblib.
# Each algorithm pipeline is run using its own instance of the :class:`~mobgap.evaluation.Evaluation` class.
#
# The evaluation object iterates over the entire dataset, runs the algorithm on each recording and calculates the
# score using the :func:`~mobgap.initial_contacts._evaluation_scorer.icd_score` function.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from mobgap.laterality.evaluation import lrc_score
from mobgap.utils.evaluation import Evaluation

n_jobs = int(get_env_var("MOBGAP_N_JOBS", 3))
results_base_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH")) / "results/lrc"
)


def run_evaluation(name, pipeline, ds):
    eval_pipe = Evaluation(
        ds,
        scoring=lrc_score,
    ).run(pipeline)
    return name, eval_pipe


# %%


def eval_debug_plot(
    results: dict[str, Evaluation[LrcEmulationPipeline]],
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
#
with Parallel(n_jobs=n_jobs) as parallel:
    results_free_living: dict[str, Evaluation[LrcEmulationPipeline]] = dict(
        parallel(
            delayed(run_evaluation)(name, pipeline, datasets_free_living)
            for name, pipeline in pipelines.items()
        )
    )

# %%
# We create a quick plot for debugging.
# This is not meant to be a comprehensive analysis, but rather a quick check to see if the results are as expected.
eval_debug_plot(results_free_living)
#
# # %%
# # Then we save the results to disk.
from mobgap.utils.evaluation import save_evaluation_results

for k, v in results_free_living.items():
    save_evaluation_results(
        k,
        v,
        condition="free_living",
        base_path=results_base_path,
        raw_results=["predictions"],
    )


# %%
# Laboratory
# ~~~~~~~~~~
# Now, we repeat the evaluation for the Laboratory part of the dataset.
with Parallel(n_jobs=n_jobs) as parallel:
    results_laboratory: dict[str, Evaluation[LrcEmulationPipeline]] = dict(
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
        raw_results=["predictions"],
    )
