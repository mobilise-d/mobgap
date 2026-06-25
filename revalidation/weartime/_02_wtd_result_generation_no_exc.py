"""
.. _wtd_val_gen_no_exc:

Revalidation of the wear-time detection algorithms
==================================================

.. note:: This script creates unpublished local validation results for the SUSTAIN wear-time dataset. The generated
    result files are not part of the published validation-result package at this time.

This script runs the wear-time detector on the SUSTAIN wear-time dataset using daily datapoints. The daily split is
important because the expected deployment mode is to apply the detector to one day at a time and because waking-hours
metrics are only well-defined for single-day datapoints.

Performance metrics are calculated on a per-day basis and aggregated over the full dataset. The raw detected
wear-time intervals, reference wear-time intervals, waking-hours reference intervals, and interval-overlap matches are
saved together with the single and aggregated score tables.

.. warning::
    Before you modify and re-run this script, read through our guide on :ref:`revalidation`.
    These results are local/unpublished; keep the script suffix ``_no_exc`` until the generated results are intended
    to be included in the public validation-result release.

"""

# %%
# Setting up the algorithms
# -------------------------
# We use the :class:`~mobgap.weartime.pipeline.WtdEmulationPipeline` to run the wear-time detector. The pipeline
# handles dataset metadata and the sensor-frame to body-frame conversion expected by the current signal-based
# detector.
from pathlib import Path

from mobgap.weartime import WtdMegaritisSignal
from mobgap.weartime.pipeline import WtdEmulationPipeline

pipelines = {
    "WtdMegaritisSignal": WtdEmulationPipeline(WtdMegaritisSignal()),
}

# %%
# Setting up the dataset
# ----------------------
# Set up your environment variables to point to the correct paths. The easiest way to do this is to create a `.env`
# file in the root of the repository with the following content. You need the path to the root folder of the SUSTAIN
# wear-time dataset `MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH` and the path where revalidation results should be stored
# `MOBGAP_VALIDATION_DATA_PATH`. The path to the cache directory `MOBGAP_CACHE_DIR_PATH` is optional.
from joblib import Memory, Parallel, delayed
from mobgap import PROJECT_ROOT
from mobgap.data import SustainWearTimeDataset
from mobgap.utils.misc import get_env_var

cache_dir = Path(get_env_var("MOBGAP_CACHE_DIR_PATH", PROJECT_ROOT / ".cache"))
results_base_path = Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH")) / "results/weartime_no_exc"
condition_name = "sustain_weartime"

dataset_sustain_weartime = SustainWearTimeDataset(
    get_env_var("MOBGAP_SUSTAIN_WEARTIME_DATASET_PATH"),
    additional_channels=(),
    split_by_day=True,
    memory=Memory(cache_dir),
)

# %%
# Running the evaluation
# ----------------------
# We multiprocess the evaluation on the level of algorithms using joblib. Each algorithm pipeline is run using its own
# instance of the :class:`~mobgap.evaluation.Evaluation` class.
#
# The scoring function returns TP/FP/FN/TN counts in samples and duration metrics in minutes.
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mobgap.utils.evaluation import Evaluation
from mobgap.weartime.evaluation import wtd_score

n_jobs = int(get_env_var("MOBGAP_N_JOBS", 3))
raw_results_to_save = ["matches", "detected", "reference", "reference_waking"]


def run_evaluation(name, pipeline, ds):
    eval_pipe = Evaluation(
        ds,
        scoring=wtd_score,
    ).run(pipeline)
    return name, eval_pipe


def eval_debug_plot(results: dict[str, Evaluation[WtdEmulationPipeline]]) -> None:
    results_df = (
        pd.concat({k: v.get_single_results_as_df() for k, v in results.items()})
        .reset_index()
        .rename(columns={"level_0": "algo_name"})
    )

    metrics = [
        "precision",
        "recall",
        "f1_score",
        "weartime_error_min",
        "waking_weartime_error_min",
        "reference_weartime_min",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 7))

    for ax, metric in zip(axes.flatten(), metrics):
        sns.boxplot(
            data=results_df,
            x="recording_type",
            y=metric,
            hue="algo_name",
            ax=ax,
            showmeans=True,
        )
        ax.set_title(metric)
        ax.set_xlabel("")

    plt.tight_layout()
    plt.show()


def save_weartime_evaluation_results(
    name: str,
    eval_obj: Evaluation[WtdEmulationPipeline],
    *,
    base_path: Path,
    condition: str,
    raw_results: list[str],
    include_non_stable_results: bool = True,
) -> None:
    folder = base_path / condition / name
    folder.mkdir(parents=True, exist_ok=True)

    raw_results_vals = eval_obj.get_raw_results()
    for key in raw_results:
        raw_results_vals[key].to_csv(folder / f"raw_{key}.csv")

    eval_obj.get_aggregated_results_as_df().drop(columns="runtime_s", errors="ignore").T.to_csv(
        folder / "aggregated_results.csv"
    )
    eval_obj.get_single_results_as_df().drop(columns="runtime_s", errors="ignore").to_csv(folder / "single_results.csv")

    if include_non_stable_results:
        with (folder / "timings.json").open("w") as file:
            json.dump(eval_obj.perf_, file, indent=2)


with Parallel(n_jobs=n_jobs) as parallel:
    results_sustain_weartime: dict[str, Evaluation[WtdEmulationPipeline]] = dict(
        parallel(
            delayed(run_evaluation)(name, pipeline, dataset_sustain_weartime)
            for name, pipeline in pipelines.items()
        )
    )

# %%
# We create a quick plot for debugging. This is not meant to be a comprehensive analysis, but rather a quick check to
# see if the generated results are plausible before writing them to disk.
eval_debug_plot(results_sustain_weartime)

# %%
# Then we save the results to disk.
for name, result in results_sustain_weartime.items():
    save_weartime_evaluation_results(
        name,
        result,
        condition=condition_name,
        base_path=results_base_path,
        raw_results=raw_results_to_save,
    )

