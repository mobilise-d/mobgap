"""
.. _pipeline_val_gen:

Revalidation of the Mobilise-D algorithm pipeline for cadence, stride length and walking speed estimation
=============================================================================================================

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
# %%
# Loading the "old" results
# -----------------------------------------------------
# Results obtained with the original Matlab-based implementation of the Mobilise-D algorithm pipeline are loaded.

from pathlib import Path
import pandas as pd
from mobgap.utils.misc import get_env_var

def load_old_fp_results(result_file_path: Path) -> pd.DataFrame:
    # A simple function to load full-pipeline results obtained with the original implementation.
    assert result_file_path.exists(), result_file_path
    per_wb_dmos_original = pd.read_csv(result_file_path).astype(
        {"participant_id": str, "start": int, "end": int}
    )
    per_wb_dmos_original = per_wb_dmos_original.set_index(per_wb_dmos_original.columns[:-4].to_list()).assign(
        ws=lambda df_: df_.avg_speed
    )
    return per_wb_dmos_original

def process_old_per_wb_results(per_wb_dmos_original: pd.DataFrame) -> pd.DataFrame:
    # Filter out Test3/Recording3 (used for calibration purposes only) #TODO: should I keep Test3/Recording3?
    if "recording" in per_wb_dmos_original.index.names: # Free-living
        grouping_var = "recording"
        per_wb_dmos_original = per_wb_dmos_original[per_wb_dmos_original.index.get_level_values("recording") != "Recording3"]
    elif "test" in per_wb_dmos_original.index.names:
        grouping_var = "test"
        per_wb_dmos_original = per_wb_dmos_original[per_wb_dmos_original.index.get_level_values("test") != "Test3"] # Laboratory
    # Compute median values per subject
    combined_eval_df = per_wb_dmos_original.groupby(["participant_id", grouping_var])[
        ["avg_speed", "avg_stride_length", "avg_cadence"]
    ].median()

    return combined_eval_df

matlab_algo_result_path = ( # Path to the folder with original results
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH"))
    / "_extracted_results/full_pipeline"
)
# Free-living original results
original_result_file_path_fl = (
    matlab_algo_result_path / "free_living" / "escience_mobilised_pipeline.csv"
)

per_wb_dmos_original_fl = load_old_fp_results(
    original_result_file_path_fl
)
# Laboratory original results

original_result_file_path_lab = (
    matlab_algo_result_path / "laboratory" / "escience_mobilised_pipeline.csv"
)

per_wb_dmos_original_lab = load_old_fp_results(
    original_result_file_path_lab
)

# Process old per-wb results
median_original_results_fl = process_old_per_wb_results(per_wb_dmos_original_fl)
median_original_results_lab = process_old_per_wb_results(per_wb_dmos_original_lab)
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
# TODO: ensure that only subjects included in Kirk et al. (2024) are loaded.

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

from mobgap.pipeline._error_metrics import ErrorTransformFuncs
from mobgap.utils.df_operations import apply_transformations
E = ErrorTransformFuncs
from matplotlib import pyplot as plt
import seaborn as sns

_errors = [
    ("walking_speed_mps", [E.error, E.abs_error, E.rel_error, E.abs_rel_error]),
    ("stride_length_m", [E.error, E.abs_error, E.rel_error, E.abs_rel_error]),
    ("cadence_spm", [E.error, E.abs_error, E.rel_error, E.abs_rel_error]),
]

def pipeline_eval_debug_plot(results: dict, median_original_results: pd.DataFrame) -> None:
    evaluation_obj = results["Official_MobiliseD_Pipeline"]

    new_results_with_errors = evaluation_obj.get_single_results_as_df()  # Extract the results_ attribute

    # Merge reference values with computed medians
    # TODO: subject 2079 has no available reference data, hence this prevents it to be included in the dataframe
    merged_df = new_results_with_errors.merge(
        median_original_results,
        left_on='participant_id',
        right_on='participant_id',
        how='inner'
    )

    # Create MultiIndex columns
    final_df = merged_df[[
        'combined__walking_speed_mps__reference', 'combined__walking_speed_mps__original',
        'combined__stride_length_m__reference', 'combined__stride_length_m__original',
        'combined__cadence_spm__reference', 'combined__cadence_spm__original'
    ]]

    final_df.columns = pd.MultiIndex.from_tuples([
        ('walking_speed_mps', 'reference'), ('walking_speed_mps', 'detected'),
        ('stride_length_m', 'reference'), ('stride_length_m', 'detected'),
        ('cadence_spm', 'reference'), ('cadence_spm', 'detected')
    ])
    original_results_errors = apply_transformations(final_df, _errors) # Calculate error metrics
    original_results_with_errors = pd.concat([original_results_errors, final_df], axis=1) # Concatenate new and original results
    original_results_with_errors.columns = ["__".join(levels) for levels in original_results_with_errors.columns]
    original_results_with_errors = original_results_with_errors.add_prefix("combined__")

    # Add a source column to distinguish between the two datasets
    new_results_with_errors['source'] = 'mobgap'
    original_results_with_errors['source'] = 'matlab'

    # Combine both datasets for plotting
    combined_df = pd.concat([new_results_with_errors, original_results_with_errors])

    # Define the metrics and outcomes of interest
    outcomes = ["walking_speed_mps", "stride_length_m", "cadence_spm"]
    metrics = ["abs_error", "abs_rel_error"]

    # Create the 2x3 boxplot figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    for col, outcome in enumerate(outcomes):
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            sns.boxplot(
                data=combined_df,
                x = "source",
                y=f"combined__{outcome}__{metric}",
                ax=ax,
                showmeans=True,
                hue="source",  # Use color to distinguish the datasets
                legend = "full"
            )
            ax.set_title(f"{metric} for {outcome}")

    plt.tight_layout()
    plt.show()

# %%
# Free-Living
# ~~~~~~~~~~~
# Let's start with the Free-Living part of the dataset.
datasets_free_living = datasets_free_living
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
pipeline_eval_debug_plot(results_free_living, median_original_results_fl)

# %%
# Then we save the results to disk.
from mobgap.utils.evaluation import save_evaluation_results

for k, v in results_free_living.items():
    save_evaluation_results(
        k,
        v,
        condition="free_living",
        base_path=results_base_path,
        raw_result_filter=["wb_level_values_with_errors"],
    )


# %%
# Laboratory
# ~~~~~~~~~~
# Now, we repeat the combined evaluation for the Laboratory part of the dataset.
with Parallel(n_jobs=n_jobs) as parallel:
    results_laboratory: dict[str, Evaluation[MobilisedPipelineUniversal]] = dict(
        parallel(
            delayed(run_evaluation)(name, pipeline, datasets_laboratory)
            for name, pipeline in pipelines.items()
        )
    )

# %%
# We create a quick plot for debugging.
pipeline_eval_debug_plot(results_laboratory, median_original_results_lab)

# %%
# Then we save the results to disk.
for k, v in results_laboratory.items():
    save_evaluation_results(
        k,
        v,
        condition="laboratory",
        base_path=results_base_path,
        raw_result_filter=["wb_level_values_with_errors"],
    )
