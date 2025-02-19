"""
.. _pipeline_val_gen:

Revalidation of the Mobilise-D computational pipeline for cadence, stride length and walking speed estimation
=============================================================================================================

.. note:: This is the code to create the results! If you are interested in viewing the results, please check the
    :ref:`results report <pipeline_val_results>`.

This script reproduces the validation results on TVS dataset for the Mobilise-D pipeline.
It loads the raw data and calculates three DMOs of interest (cadence, stride length and walking speed).
Performance metrics are calculated on a per-trial/per-recording basis and aggregated (median for most metrics)
over the whole dataset.
The raw per second cadence, stride length, walking speed and all performance metrics are saved to disk.

.. warning:: Before you modify and re-run this script, read through our guide on :ref:`revalidation`.
   In case you are planning to update the official results (either after a code change, or because an algorithm was
   added), contact one of the core maintainers.
   They can assist with the process.

"""
from pathlib import Path
from mobgap.pipeline import MobilisedPipelineHealthy, MobilisedPipelineImpaired, MobilisedPipelineUniversal
import pandas as pd

def load_old_fp_results(result_file_path: Path) -> pd.DataFrame:
    assert result_file_path.exists(), result_file_path
    data = pd.read_csv(result_file_path).astype(
        {"participant_id": str, "start": int, "end": int}
    )
    data = data.set_index(data.columns[:-4].to_list()).assign(
        sl_per_sec=lambda df_: df_.avg_speed
    )
    return data
# %%
# Setting up the algorithms
# -------------------------
# We use the :class:`~mobgap.pipeline.MobilisedPipelineUniversal` to run the algorithms.
# We create an instance of this pipeline for each DMO we want to evaluate and store them in a dictionary.
# The key is used to identify the algorithm in the results and used as folder name to store the results.
#
# .. note:: Set up your environment variables to point to the correct paths.
#    The easiest way to do this is to create a `.env` file in the root of the repository with the following content.
#    You need the paths to the root folder of the TVS dataset `MOBGAP_TVS_DATASET_PATH` and the path where revalidation
#    results should be stored `MOBGAP_VALIDATION_DATA_PATH`.
#    The path to the cache directory `MOBGAP_CACHE_DIR_PATH` is optional, when you don't want to store the memory cache
#    in the default location.
from mobgap.utils.misc import get_env_var

matlab_algo_result_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH")) / "_extracted_results/full_pipeline"
)
base_result_folder=matlab_algo_result_path
measurement_condition = "free_living"
matlab_algo_name = "escience_mobilised_pipeline"
result_file_path = base_result_folder / measurement_condition/ f"{matlab_algo_name}.csv"

old_results = load_old_fp_results(result_file_path) # not necessary for this example

# Define a universal pipeline object including the two pipelines (healthy and impaired)
pipelines = {}
pipelines["Official_MobiliseD_Pipeline"] = MobilisedPipelineUniversal(
    pipelines=[
        ("healthy", MobilisedPipelineHealthy()),
        ("impaired", MobilisedPipelineImpaired()),
    ]
)
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
from joblib import Memory, Parallel, delayed
from mobgap import PACKAGE_ROOT
from mobgap.data import TVSFreeLivingDataset, TVSLabDataset

cache_dir = Path(
    get_env_var("MOBGAP_CACHE_DIR_PATH", PACKAGE_ROOT.parent / ".cache")
)

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
# ------------- ---------
# We multiprocess the evaluation on the level of algorithms using joblib.
# Each algorithm pipeline is run using its own instance of the :class:`~mobgap.evaluation.Evaluation` class.
#
# The evaluation object iterates over the entire dataset, runs the algorithm on each recording and calculates the
# score using the :func:`~mobgap.gait_sequences._evaluation_scorer.gsd_score` function.

from pathlib import Path

import pandas as pd
from mobgap.pipeline.evaluation import pipeline_score
from mobgap.utils.evaluation import Evaluation

n_jobs = int(get_env_var("MOBGAP_N_JOBS", 3))
results_base_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH")) / "results/full_pipeline"
)


def run_evaluation(name, pipeline, ds):
    eval_pipe = Evaluation(
        ds,
        scoring=pipeline_score,
    ).run(pipeline)
    return name, eval_pipe



# %%
# Free-Living
# ~~~~~~~~~~~
# Let's start with the Free-Living part of the dataset.
with Parallel(n_jobs=n_jobs) as parallel:
    results_free_living: dict[str, Evaluation[MobilisedPipelineUniversal]] = dict(
        parallel(
            delayed(run_evaluation)(name, pipeline, datasets_free_living)
            for name, pipeline in pipelines.items()
        )
    )
results_free_living
# %%
# We create a quick plot for debugging.
# This is not meant to be a comprehensive analysis, but rather a quick check to see if the results are as expected.
#
# Note, that wb-level means that each datapoint used to create the results is a single walking bout.
# Measurement-level means that each datapoint is a single recording/participant.
# The value error value per participant was itself calculated as the mean of the error values of all walking bouts of
# that participant.
# eval_debug_plot(results_free_living)

# %%
# Then we save the results to disk.
from mobgap.utils.evaluation import save_evaluation_results

for k, v in results_free_living.items():
    save_evaluation_results(
        k,
        v,
        condition="free_living",
        base_path=results_base_path,
        raw_result_filter=["wb_level_ws_values_with_errors"],
    )


# # %%
# # Laboratory
# # ~~~~~~~~~~
# # Now, we repeat the evaluation for the Laboratory part of the dataset.
# with Parallel(n_jobs=n_jobs) as parallel:
#     results_laboratory: dict[str, Evaluation[SlEmulationPipeline]] = dict(
#         parallel(
#             delayed(run_evaluation)(name, pipeline, datasets_laboratory)
#             for name, pipeline in pipelines.items()
#         )
#     )
#
# # %%
# # We create a quick plot for debugging.
# eval_debug_plot(results_laboratory)
#
# # %%
# # Then we save the results to disk.
# for k, v in results_laboratory.items():
#     save_evaluation_results(
#         k,
#         v,
#         condition="laboratory",
#         base_path=results_base_path,
#         raw_result_filter=["wb_level_values_with_errors"],
#     )
