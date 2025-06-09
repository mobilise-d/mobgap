"""
.. _cad_val_gen:

Revalidation of the cadence algorithms
======================================

.. note:: This is the code to create the results! If you are interested in viewing the results, please check the
    :ref:`results report <cad_val_results>`.

This script reproduces the validation results on TVS dataset for the cadence.
It load results from the old matlab algorithms and runs the new algorithms on the same data.
Performance metrics are calculated on a per-trial/per-recording basis and aggregated (mean for most metrics)
over the whole dataset.
The raw per second cadence and all performance metrics are saved to disk.

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
from mobgap.cadence import CadFromIcDetector
from mobgap.cadence.base import BaseCadCalculator
from mobgap.cadence.evaluation import cad_score
from mobgap.cadence.pipeline import CadEmulationPipeline
from mobgap.initial_contacts import IcdHKLeeImproved, IcdShinImproved
from mobgap.pipeline import Region
from mobgap.utils.conversions import as_samples
from tpcp.caching import hybrid_cache
from typing_extensions import Self, Unpack


def _process_cad_sec(unparsed: str) -> list[float]:
    """Process the cadence per second from the matlab file."""
    # We parse the sting manually to handle the "nan" values.
    unparsed = unparsed.strip("[]")
    parts = unparsed.split(", ")
    return [np.nan if x == "nan" else float(x) for x in parts]


def load_old_cad_results(result_file_path: Path) -> pd.DataFrame:
    assert result_file_path.exists(), result_file_path
    data = pd.read_csv(result_file_path).astype(
        {"participant_id": str, "start": int, "end": int}
    )
    data = data.set_index(data.columns[:-4].to_list()).assign(
        cad_per_sec=lambda df_: df_.cad_per_sec.map(_process_cad_sec)
    )
    return data


class DummyCadAlgo(BaseCadCalculator):
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

        cached_load_old_cad_results = hybrid_cache(lru_cache_maxsize=1)(
            load_old_cad_results
        )

        all_results = cached_load_old_cad_results(
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
            cad_per_sec = np.full_like(sec_centers, np.nan)
            index = pd.Index(
                as_samples(sec_centers, sampling_rate_hz),
                name="sec_center_samples",
            )
            self.cadence_per_sec_ = pd.DataFrame(
                {"cadence_spm": cad_per_sec}, index=index
            )
            return self

        gs_start = current_gs_absolute.start
        gs_end = current_gs_absolute.end

        # We need to fuzzy search for the start, as rounding was done differently in the old pipeline.
        cad_results = recording_results[
            recording_results.start.isin([gs_start, gs_start + 1, gs_start - 1])
        ]
        if len(cad_results) == 0:
            raise ValueError(
                f"No results found for {dp_group}, {current_gs_absolute}"
            )
        if len(cad_results) > 1:
            raise ValueError(
                f"Multiple results found for {dp_group}, {current_gs_absolute}"
            )
        if cad_results.iloc[0].end not in [gs_end, gs_end - 1, gs_end + 1]:
            raise ValueError(
                f"End does not match for {dp_group}, {current_gs_absolute}"
            )
        cad_per_sec = cad_results["cad_per_sec"].iloc[0]

        # The number of second in the old algorithms is sometimes different then for the new algorithms.
        # In the new pipeline, we extrapolate slightly to ensure that we cover the full duration of the GS.
        # In the old pipeline, only "full" seconds were used.
        # So the old pipeline result could be 1 value shorter than the new pipeline result.
        # In this case we replicate the last value.
        # This is similar to what we do in the new pipeline.

        if len(cad_per_sec) == len(sec_centers) - 1:
            cad_per_sec = np.concatenate((cad_per_sec, [cad_per_sec[-1]]))

        if len(cad_per_sec) != len(sec_centers):
            raise ValueError(
                f"Length mismatch between cadence per second and sec_centers: {len(cad_per_sec)} != {len(sec_centers)} "
                "We assume that the results of the old pipeline either have the same number of seconds, or 1 value less."
            )

        index = pd.Index(
            as_samples(sec_centers, sampling_rate_hz), name="sec_center_samples"
        )
        self.cadence_per_sec_ = pd.DataFrame(
            {"cadence_spm": cad_per_sec}, index=index
        )
        return self


# %%
# Setting up the algorithms
# -------------------------
# We use the :class:`~mobgap.cadence.pipeline.CadEmulationPipeline` to run the algorithms.
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
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH")) / "_extracted_results/cad"
)

pipelines = {}
for matlab_algo_name in [
    "HKLee_Imp2",
    "Shin_Imp",
]:
    pipelines[f"matlab_{matlab_algo_name}"] = CadEmulationPipeline(
        DummyCadAlgo(
            matlab_algo_name, base_result_folder=matlab_algo_result_path
        )
    )
pipelines["HKLeeImproved"] = CadEmulationPipeline(
    CadFromIcDetector(IcdHKLeeImproved())
)
pipelines["ShinImproved"] = CadEmulationPipeline(
    CadFromIcDetector(IcdShinImproved())
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
# score using the :func:`~mobgap.cadence.evaluation.cad_score` function.
import seaborn as sns
from mobgap.utils.evaluation import Evaluation

n_jobs = int(get_env_var("MOBGAP_N_JOBS", 3))
results_base_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH")) / "results/cad"
)


def run_evaluation(name, pipeline, ds):
    eval_pipe = Evaluation(
        ds,
        scoring=cad_score,
    ).run(pipeline)
    return name, eval_pipe


def eval_debug_plot(
    results: dict[str, Evaluation[CadEmulationPipeline]],
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
    results_free_living: dict[str, Evaluation[CadEmulationPipeline]] = dict(
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
    results_laboratory: dict[str, Evaluation[CadEmulationPipeline]] = dict(
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
