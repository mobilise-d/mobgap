"""
.. _sl_val_gen:

Revalidation of the stride length algorithms
============================================

.. note:: This is the code to create the results! If you are interested in viewing the results, please check the
    :ref:`results report <sl_val_results>`.

This script reproduces the validation results on TVS dataset for the stride length.
It load results from the old matlab algorithms and runs the new algorithms on the same data.
Performance metrics are calculated on a per-trial/per-recording basis and aggregated (mean for most metrics)
over the whole dataset.
The raw per second stride length and all performance metrics are saved to disk.

.. warning:: Before you modify and re-run this script, read through our guide on :ref:`revalidation`.
   In case you are planning to update the official results (either after a code change, or because an algorithm was
   added), contact one of the core maintainers.
   They can assist with the process.

"""

# %%
# Setting up "Dummy Algorithms" to load the old results
# -----------------------------------------------------
# Instead of just loading the results of the old matlab algorithms, we create dummy algorithms that respond with the
# precomputed results per trial.
# This way, we can be sure that the exact same structure is used for the evaluation of all algorithms.
#
# Note, that this is not the most efficient way to do this, as we need to open the file repeatedly and also reload the
# data from the matlab files, even though the dummy algorithm does not need it.
import warnings
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mobgap.pipeline import Region
from mobgap.stride_length import SlZijlstra
from mobgap.stride_length.base import BaseSlCalculator
from mobgap.stride_length.evaluation import sl_score
from mobgap.stride_length.pipeline import SlEmulationPipeline
from mobgap.utils.conversions import as_samples
from tpcp.caching import hybrid_cache
from typing_extensions import Self, Unpack


def _process_sl_sec(unparsed: str) -> list[float]:
    """Process the stride length per second from the matlab file."""
    # We parse the sting manually to handle the "nan" values.
    unparsed = unparsed.strip("[]")
    parts = unparsed.split(", ")
    return [np.nan if x == "nan" else float(x) for x in parts]


def load_old_sl_results(result_file_path: Path) -> pd.DataFrame:
    assert result_file_path.exists(), result_file_path
    data = pd.read_csv(result_file_path).astype(
        {"participant_id": str, "start": int, "end": int}
    )
    data = data.set_index(data.columns[:-4].to_list()).assign(
        sl_per_sec=lambda df_: df_.sl_per_sec.map(_process_sl_sec)
    )
    return data


class DummySlAlgo(BaseSlCalculator):
    """A dummy algorithm that responds with the precomputed results of the old pipeline.

    This makes it convenient to compare the results of the old pipeline with the new pipeline, as we can simply use
    the same code to evaluate both.
    However, this also makes things a lot slower compared to just loading all the results, as we need to open the
    file repeatedly and also reload the data from the matlab files, even though the dummy algorithm does not need it.

    Parameters
    ----------
    old_algo_name
        Name of the algorithm for which we want to load the results.
        This determines the name of the file to load.
    base_result_folder
        Base folder where the results are stored.

    """

    def __init__(self, old_algo_name: str, base_result_folder: Path) -> None:
        self.old_algo_name = old_algo_name
        self.base_result_folder = base_result_folder

    def calculate(
        self,
        data: pd.DataFrame,
        initial_contacts: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        measurement_condition: Optional[
            Literal["free_living", "laboratory"]
        ] = None,
        dp_group: Optional[tuple[str, ...]] = None,
        current_gs_absolute: Optional[Region] = None,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """ "Run" the algorithm."""
        assert measurement_condition is not None, (
            "measurement_condition must be provided"
        )
        assert dp_group is not None, "dp_group must be provided"
        assert current_gs_absolute is not None, (
            "current_gs_start_absolute must be provided"
        )

        cached_load_old_sl_results = hybrid_cache(lru_cache_maxsize=1)(
            load_old_sl_results
        )

        all_results = cached_load_old_sl_results(
            self.base_result_folder
            / measurement_condition
            / f"{self.old_algo_name}.csv"
        )

        unique_label = dp_group[:-2]
        duration = data.shape[0] / sampling_rate_hz
        sec_centers = np.arange(0, duration) + 0.5

        try:
            recording_results = all_results.loc[unique_label]
        except KeyError:
            warnings.warn(
                f"No result found for recording {unique_label}. "
                "We will replace results with NaNs.",
                RuntimeWarning,
            )
            sl_per_sec = np.full_like(sec_centers, np.nan)
            index = pd.Index(
                as_samples(sec_centers, sampling_rate_hz),
                name="sec_center_samples",
            )
            self.stride_length_per_sec_ = pd.DataFrame(
                {"stride_length_m": sl_per_sec}, index=index
            )
            return self

        gs_start = current_gs_absolute.start
        gs_end = current_gs_absolute.end

        # We need to fuzzy search for the start, as rounding was done differently in the old pipeline.
        sl_results = recording_results[
            recording_results.start.isin([gs_start, gs_start + 1, gs_start - 1])
        ]
        if len(sl_results) == 0:
            raise ValueError(
                f"No results found for {dp_group}, {current_gs_absolute}"
            )
        if len(sl_results) > 1:
            raise ValueError(
                f"Multiple results found for {dp_group}, {current_gs_absolute}"
            )
        if sl_results.iloc[0].end not in [gs_end, gs_end - 1, gs_end + 1]:
            raise ValueError(
                f"End does not match for {dp_group}, {current_gs_absolute}"
            )
        sl_per_sec = sl_results["sl_per_sec"].iloc[0]

        # The number of second in the old algorithms is sometimes different then for the new algorithms.
        # In the new pipeline, we extrapolate slightly to ensure that we cover the full duration of the GS.
        # In the old pipeline, only "full" seconds were used.
        # So the old pipeline result could be 1 value shorter than the new pipeline result.
        # In this case we replicate the last value.
        # This is similar to what we do in the new pipeline.

        if len(sl_per_sec) == len(sec_centers) - 1:
            sl_per_sec = np.concatenate((sl_per_sec, [sl_per_sec[-1]]))

        if len(sl_per_sec) != len(sec_centers):
            raise ValueError(
                f"Length mismatch between stride length per second and sec_centers: {len(sl_per_sec)} != {len(sec_centers)} "
                "We assume that the results of the old pipeline either have the same number of seconds, or 1 value less."
            )

        index = pd.Index(
            as_samples(sec_centers, sampling_rate_hz), name="sec_center_samples"
        )
        self.stride_length_per_sec_ = pd.DataFrame(
            {"stride_length_m": sl_per_sec}, index=index
        )
        return self


# %%
# Setting up the algorithms
# -------------------------
# We use the :class:`~mobgap.gait_sequences.pipeline.GsdEmulationPipeline` to run the algorithms.
# We create an instance of this pipeline for each algorithm we want to evaluate and store them in a dictionary.
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
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH")) / "_extracted_results/sl"
)

pipelines = {}
for matlab_algo_name in [
    "zjilsV3__MS_ALL",
    "zjilsV3__MS_MS",
]:
    pipelines[f"matlab_{matlab_algo_name}"] = SlEmulationPipeline(
        DummySlAlgo(
            matlab_algo_name, base_result_folder=matlab_algo_result_path
        )
    )
pipelines["SlZjilstra__MS_ALL"] = SlEmulationPipeline(
    SlZijlstra(
        **SlZijlstra.PredefinedParameters.step_length_scaling_factor_ms_all
    )
)
pipelines["SlZjilstra__MS_MS"] = SlEmulationPipeline(
    SlZijlstra(
        **SlZijlstra.PredefinedParameters.step_length_scaling_factor_ms_ms
    )
)

# %%
# For the reimplemented algorithm, we set up version with different default presets.


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
# score using the :func:`~mobgap.gait_sequences._evaluation_scorer.gsd_score` function.
import seaborn as sns
from mobgap.utils.evaluation import Evaluation

n_jobs = int(get_env_var("MOBGAP_N_JOBS", 3))
results_base_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH")) / "results/sl"
)


def run_evaluation(name, pipeline, ds):
    eval_pipe = Evaluation(
        ds,
        scoring=sl_score,
    ).run(pipeline)
    return name, eval_pipe


def eval_debug_plot(
    results: dict[str, Evaluation[SlEmulationPipeline]],
) -> None:
    results_df_wb = (
        pd.concat(
            {
                k: v.get_raw_results()["wb_level_values_with_errors"]
                for k, v in results.items()
            }
        )
        .reset_index()
        .rename(columns={"level_0": "algo_name"})
    )
    results_df_measurement = (
        pd.concat({k: v.get_single_results_as_df() for k, v in results.items()})
        .filter(like="wb__")
        .rename(columns=lambda x: x.strip("wb__"))
        .reset_index()
        .rename(columns={"level_0": "algo_name"})
    )

    metrics = [
        "error",
        "abs_error",
        "abs_rel_error",
    ]
    results = [
        ("wb_level", results_df_wb),
        ("measurement_level", results_df_measurement),
    ]
    combinations = [(n, c, m) for n, c in results for m in metrics]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    for ax, (n, c, m) in zip(axes.flatten(), combinations):
        sns.boxplot(
            data=c,
            x="cohort",
            y=m,
            hue="algo_name",
            ax=ax,
            showmeans=True,
        )
        ax.set_title(f"{m} {n}")

    plt.tight_layout()
    plt.show()


# %%
# Free-Living
# ~~~~~~~~~~~
# Let's start with the Free-Living part of the dataset.

with Parallel(n_jobs=n_jobs) as parallel:
    results_free_living: dict[str, Evaluation[SlEmulationPipeline]] = dict(
        parallel(
            delayed(run_evaluation)(name, pipeline, datasets_free_living)
            for name, pipeline in pipelines.items()
        )
    )

# %%
# We create a quick plot for debugging.
# This is not meant to be a comprehensive analysis, but rather a quick check to see if the results are as expected.
#
# Note, that wb-level means that each datapoint used to create the results is a single walking bout.
# Measurement-level means that each datapoint is a single recording/participant.
# The value error value per participant was itself calculated as the mean of the error values of all walking bouts of
# that participant.
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
        raw_results=["wb_level_values_with_errors"],
    )

# %%
# Laboratory
# ~~~~~~~~~~
# Now, we repeat the evaluation for the Laboratory part of the dataset.
with Parallel(n_jobs=n_jobs) as parallel:
    results_laboratory: dict[str, Evaluation[SlEmulationPipeline]] = dict(
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
        raw_results=["wb_level_values_with_errors"],
    )
