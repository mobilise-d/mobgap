"""
.. _gsd_val_gen:

Revalidation of the gait sequence detection algorithms
======================================================

.. note:: This is the code to create the results! If you are interested in viewing the results, please check the
    :ref:`results report <gsd_val_results>`.

This script reproduces the validation results on TVS dataset for the gait sequence detection algorithms.
It load results from the old matlab algorithms and runs the new algorithms on the same data.
Performance metrics are calculated on a per-trial/per-recording basis and aggregated (mean for most metrics)
over the whole dataset.
The raw detected gait sequences and all performance metrics are saved to disk.

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
#
# As part of these dummy algorithms, we also clip the GSDs to the length of the data, as some algorithms provide GSDs
# that extend past the end of the data.
# In most cases, these are rounding issues.
# In case of the EPFL_V1-* algorithms, this is caused by an actual bug in the original implementation.
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from mobgap.gait_sequences.base import BaseGsDetector
from tpcp.caching import hybrid_cache
from typing_extensions import Self, Unpack


def load_old_gsd_results(result_file_path: Path) -> pd.DataFrame:
    assert result_file_path.exists(), result_file_path
    data = pd.read_csv(result_file_path).astype({"participant_id": str})
    data = data.set_index(data.columns[:-2].to_list())
    return data


class DummyGsdAlgo(BaseGsDetector):
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

    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        measurement_condition: Optional[
            Literal["free_living", "laboratory"]
        ] = None,
        dp_group: Optional[tuple[str, ...]] = None,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """ "Run" the algorithm."""
        assert (
            measurement_condition is not None
        ), "measurement_condition must be provided"
        assert dp_group is not None, "dp_group must be provided"

        cached_load_old_gsd_results = hybrid_cache(lru_cache_maxsize=1)(
            load_old_gsd_results
        )

        all_results = cached_load_old_gsd_results(
            self.base_result_folder
            / measurement_condition
            / f"{self.old_algo_name}.csv"
        )

        unique_label = dp_group[:-2]
        try:
            gs_list = all_results.loc[unique_label].copy()
        except KeyError:
            gs_list = pd.DataFrame(columns=["start", "end"]).astype(
                {"start": int, "end": int}
            )

        # For some reason some algorithms provide start and end values that are larger than the length of the signal.
        # We clip them here.
        gs_list["end"] = gs_list["end"].clip(upper=len(data))
        # And then we remove the ones where the end is smaller or equal to start.
        gs_list = gs_list[gs_list["start"] < gs_list["end"]]

        self.gs_list_ = gs_list.rename_axis("gs_id").copy()
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
from mobgap.gait_sequences.pipeline import GsdEmulationPipeline
from mobgap.utils.misc import get_env_var

matlab_algo_result_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH")) / "_extracted_results/gsd"
)

pipelines = {}
for matlab_algo_name in [
    "EPFL_V1-improved_th",
    "EPFL_V1-original",
    "EPFL_V2-original",
    # "Gaitpy",
    # "Hickey-original",
    # "Rai",
    "TA_Iluz-original",
    # "TA_Wavelets_v2",
]:
    pipelines[f"matlab_{matlab_algo_name}"] = GsdEmulationPipeline(
        DummyGsdAlgo(
            matlab_algo_name, base_result_folder=matlab_algo_result_path
        )
    )

# %%
# For the reimplemented algorithm, we set up version with different default presets.
from mobgap.gait_sequences import GsdAdaptiveIonescu, GsdIluz, GsdIonescu

pipelines["GsdIluz"] = GsdEmulationPipeline(GsdIluz())
pipelines["GsdIluz_orig_peak"] = GsdEmulationPipeline(
    GsdIluz(**GsdIluz.PredefinedParameters.original)
)
pipelines["GsdIonescu"] = GsdEmulationPipeline(GsdIonescu())
pipelines["GsdAdaptiveIonescu"] = GsdEmulationPipeline(GsdAdaptiveIonescu())

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
from joblib import Memory
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
# ----------------------
# We multiprocess the evaluation on the level of algorithms using joblib.
# Each algorithm pipeline is run using its own instance of the :class:`~mobgap.evaluation.Evaluation` class.
#
# The evaluation object iterates over the entire dataset, runs the algorithm on each recording and calculates the
# score using the :func:`~mobgap.gait_sequences._evaluation_scorer.gsd_score` function.
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from mobgap.gait_sequences.evaluation import gsd_score
from mobgap.utils.evaluation import Evaluation

n_jobs = int(get_env_var("MOBGAP_N_JOBS", 3))
results_base_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH")) / "results/gsd"
)


def run_evaluation(name, pipeline, ds):
    eval_pipe = Evaluation(
        ds,
        scoring=gsd_score,
    ).run(pipeline)
    return name, eval_pipe


def eval_debug_plot(
    results: dict[str, Evaluation[GsdEmulationPipeline]],
) -> None:
    results_df = (
        pd.concat({k: v.get_single_results_as_df() for k, v in results.items()})
        .reset_index()
        .rename(columns={"level_0": "algo_name"})
    )

    metrics = [
        "precision",
        "recall",
        "f1_score",
        "accuracy",
        "gs_duration_error_s",
        "gs_relative_duration_error",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 6))

    for ax, metric in zip(axes.flatten(), metrics):
        sns.boxplot(
            data=results_df,
            x="cohort",
            y=metric,
            hue="algo_name",
            ax=ax,
            showmeans=True,
        )
        ax.set_title(metric)

    plt.tight_layout()
    plt.show()


# %%
# Free-Living
# ~~~~~~~~~~~
# Let's start with the Free-Living part of the dataset.

with Parallel(n_jobs=n_jobs) as parallel:
    results_free_living: dict[str, Evaluation[GsdEmulationPipeline]] = dict(
        parallel(
            delayed(run_evaluation)(name, pipeline, datasets_free_living)
            for name, pipeline in pipelines.items()
        )
    )

# %%
# We create a quick plot for debugging.
# This is not meant to be a comprehensive analysis, but rather a quick check to see if the results are as expected.
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
        raw_result_filter=["detected"],
    )


# %%
# Laboratory
# ~~~~~~~~~~
# Now, we repeat the evaluation for the Laboratory part of the dataset.
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
        raw_result_filter=["detected"],
    )
