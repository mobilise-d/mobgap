"""
.. _pipeline_val_gen:

Revalidation of the Mobilise-D algorithm pipeline for cadence, stride length and walking speed estimation
=========================================================================================================

.. note:: This is the code to create the results! If you are interested in viewing the results, please check the
    :ref:`results report <pipeline_val_results>`.

This script reproduces the validation results on TVS dataset for the Mobilise-D algorithm pipeline.
It loads the raw data and calculates three DMOs of interest (cadence, stride length and walking speed) using the full
pipeline, including the following blocks:
    1) gait_sequence_detection
    2) initial_contact_detection
    3) laterality_classification
    4) cadence_calculation
    5) stride_length_calculation
    6) walking_speed_calculation
    7) turn_detection
    8) stride_selection
    9) wba
    10) dmo_thresholds
    11) dmo_aggregation
Performance metrics are calculated on a per-trial/per-recording basis and aggregated (median for most metrics)
over the whole dataset.
The raw per second cadence, stride length, walking speed and all performance metrics are saved to disk.

.. warning:: Before you modify and re-run this script, read through our guide on :ref:`revalidation`.
   In case you are planning to update the official results (either after a code change, or because an algorithm was
   added), contact one of the core maintainers.
   They can assist with the process.

"""

import warnings

# %%
# Loading the "old" results
# -------------------------
# Results obtained with the original Matlab-based implementation of the Mobilise-D algorithm pipeline are loaded.
# We wrap these results in a dummy pipeline that acts like the real pipeline, but simply returns the pre-calculated
# results.
# This way, we can ensure that the exact format, order and participants are used for the comparison.
from pathlib import Path
from typing import Optional, Self

import pandas as pd
from mobgap.data import BaseTVSDataset, TVSFreeLivingDataset, TVSLabDataset
from mobgap.laterality import LrcUllrich
from mobgap.pipeline.base import BaseMobilisedPipeline
from mobgap.utils.misc import get_env_var
from tpcp.caching import hybrid_cache

from revalidation.gait_sequences import DummyGsdAlgo


def load_old_fp_results(result_file_path: Path) -> pd.DataFrame:
    # A simple function to load full-pipeline results obtained with the original implementation.
    assert result_file_path.exists(), result_file_path
    per_wb_dmos_original = pd.read_csv(result_file_path).astype(
        {"participant_id": str, "start": int, "end": int}
    )
    if "recording" in per_wb_dmos_original.columns:
        index_cols = [
            "cohort",
            "participant_id",
            "time_measure",
            "recording",
            "wb_id",
        ]
    elif "test" in per_wb_dmos_original.columns:
        index_cols = [
            "cohort",
            "participant_id",
            "time_measure",
            "test",
            "trial",
            "wb_id",
        ]
    else:
        raise ValueError("Could not determine the index columns.")

    per_wb_dmos_original = (
        per_wb_dmos_original.set_index(index_cols)
        .rename(
            columns={
                "avg_cadence": "cadence_spm",
                "avg_stride_length": "stride_length_m",
                "avg_stride_duration": "stride_duration_s",
                "avg_speed": "walking_speed_mps",
                "duration_s": "duration_s",
            }
        )
        .drop(
            columns=["start_datetime_utc", "start_timestamp_utc", "time_zone"]
        )
    )
    return per_wb_dmos_original


class DummyFullPipeline(BaseMobilisedPipeline[BaseTVSDataset]):
    def __init__(self, result_file_path: Path) -> None:
        self.result_file_path = result_file_path

    def get_recommended_cohorts(self) -> Optional[tuple[str, ...]]:
        return MobilisedPipelineUniversal().get_recommended_cohorts()

    def run(self, datapoint: BaseTVSDataset) -> Self:
        cached_load_old_fp_results = hybrid_cache(lru_cache_maxsize=1)(
            load_old_fp_results
        )

        old_results = cached_load_old_fp_results(
            self.result_file_path
            / datapoint.recording_metadata["measurement_condition"]
            / "escience_mobilised_pipeline.csv"
        )

        n_relevant_index_cols = (
            4 if "recording" in old_results.index.names else 5
        )

        try:
            per_wb_results = old_results.loc[
                datapoint.group_label[:n_relevant_index_cols]
            ]
        except KeyError:
            warnings.warn(f"No results found for {datapoint.group_label}.")
            per_wb_results = pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "walking_speed_mps",
                    "stride_length_m",
                    "cadence_spm",
                ]
            )
        self.per_wb_parameters_ = per_wb_results
        return self


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
from mobgap.pipeline import (
    MobilisedPipelineHealthy,
    MobilisedPipelineImpaired,
    MobilisedPipelineUniversal,
)

escience_pipeline_result_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH"))
    / "_extracted_results/full_pipeline"
)
escience_pipeline_result_path_gsd = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH")) / "_extracted_results/gsd"
)

# Define a universal pipeline object including the two pipelines (healthy and impaired)
pipelines = {
    "Official_MobiliseD_Pipeline": MobilisedPipelineUniversal(),
    "Official_MobiliseD_Pipeline__old_gs": MobilisedPipelineUniversal(
        pipelines=[
            (
                "healthy",
                MobilisedPipelineHealthy(
                    gait_sequence_detection=DummyGsdAlgo(
                        "EPFL_V1-improved_th",
                        escience_pipeline_result_path_gsd,
                        min_gs_duration_s=3,
                    )
                ),
            ),
            (
                "impaired",
                MobilisedPipelineImpaired(
                    gait_sequence_detection=DummyGsdAlgo(
                        "EPFL_V1-improved_th",
                        escience_pipeline_result_path_gsd,
                        min_gs_duration_s=3,
                    )
                ),
            ),
        ]
    ),
    "Official_MobiliseD_Pipeline__old_lrc": MobilisedPipelineUniversal(
        pipelines=[
            (
                "healthy",
                MobilisedPipelineHealthy(
                    laterality_classification=LrcUllrich(
                        **LrcUllrich.PredefinedParameters.msproject_all_old
                    )
                ),
            ),
            (
                "impaired",
                MobilisedPipelineImpaired(
                    laterality_classification=LrcUllrich(
                        **LrcUllrich.PredefinedParameters.msproject_all_old
                    )
                ),
            ),
        ]
    ),
    "EScience_MobiliseD_Pipeline": DummyFullPipeline(
        escience_pipeline_result_path
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

from joblib import Memory, Parallel, delayed
from mobgap import PROJECT_ROOT

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
# score using the :func:`~mobgap.gait_sequences._evaluation_scorer.gsd_score` function.
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


import seaborn as sns
from matplotlib import pyplot as plt


def pipeline_eval_debug_plot(
    results: dict[str, Evaluation[BaseMobilisedPipeline]],
) -> None:
    results_df_wb = (
        pd.concat({k: v.get_single_results_as_df() for k, v in results.items()})
        .reset_index()
        .rename(columns={"level_0": "algo_name"})
    )

    # Define the metrics and outcomes of interest
    outcomes = ["walking_speed_mps", "stride_length_m", "cadence_spm"]
    metrics = ["error", "abs_error", "abs_rel_error"]

    # Create the 3x3 boxplot figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for col, outcome in enumerate(outcomes):
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            sns.boxplot(
                data=results_df_wb,
                x="algo_name",
                y=f"combined__{outcome}__{metric}",
                ax=ax,
                showmeans=True,
                hue="algo_name",
                legend=False,
            )
            ax.set_title(f"{metric} for {outcome}")

    plt.tight_layout()
    plt.show()


# %%
# Free-Living
# ~~~~~~~~~~~
# Let's start with the Free-Living part of the dataset.
with Parallel(n_jobs=n_jobs) as parallel:
    results_free_living: dict[str, Evaluation[MobilisedPipelineUniversal]] = (
        dict(
            parallel(
                delayed(run_evaluation)(name, pipeline, datasets_free_living)
                for name, pipeline in pipelines.items()
            )
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
pipeline_eval_debug_plot(results_free_living)

# %%
# Then we save the results to disk.
from mobgap.utils.evaluation import save_evaluation_results

for k, v in results_free_living.items():
    save_evaluation_results(
        k,
        v,
        condition="free_living",
        base_path=results_base_path,
        raw_results=["matched_errors"],
    )


# %%
# Laboratory
# ~~~~~~~~~~
# Now, we repeat the combined evaluation for the Laboratory part of the dataset.
with Parallel(n_jobs=n_jobs) as parallel:
    results_laboratory: dict[str, Evaluation[MobilisedPipelineUniversal]] = (
        dict(
            parallel(
                delayed(run_evaluation)(name, pipeline, datasets_laboratory)
                for name, pipeline in pipelines.items()
            )
        )
    )

# %%
# We create a quick plot for debugging.
pipeline_eval_debug_plot(results_laboratory)

# %%
# Then we save the results to disk.
for k, v in results_laboratory.items():
    save_evaluation_results(
        k,
        v,
        condition="laboratory",
        base_path=results_base_path,
        raw_results=["matched_errors"],
    )
