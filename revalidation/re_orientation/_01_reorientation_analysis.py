"""
.. _reorientation_val_results:

Performance of the reorientation algorithm on simulated TVS misorientations
===========================================================================

The TVS dataset does not provide recordings with known lower-back sensor
misorientations. For this revalidation, we therefore use the INDIP reference
walking bouts, simulate all supported rough mounting orientations, and treat the
algorithm response as a multiclass classification problem.

We compare the full and conservative variants of the
:class:`~mobgap.re_orientation.ReorientationMethodDM` algorithm.

.. note:: If you are interested in how these results are calculated, head over to
    the :ref:`processing page <reorientation_val_gen>`.

"""

# %%
# Algorithms
# ----------
# The result generation script stores one result folder per algorithm variant.
# Here, we map these folder names to display labels used in plots and tables.
algorithms = {
    "MethodDM__full": ("ReorientationMethodDM", "Full"),
    "MethodDM__conservative": ("ReorientationMethodDM", "Conservative"),
}

# %%
# Loading Results
# ---------------
# By default, the data will be downloaded from the validation result repository.
# During development, set `MOBGAP_VALIDATION_USE_LOCAL_DATA=1` and point
# `MOBGAP_VALIDATION_DATA_PATH` to the local validation-data folder.
from pathlib import Path

import pandas as pd
from mobgap.data.validation_results import ValidationResultLoader
from mobgap.re_orientation.pipeline import REORIENTATION_LABELS
from mobgap.utils.misc import get_env_var


def format_loaded_results(
    values: dict[tuple[str, str], pd.DataFrame],
    index_cols: list[str],
) -> pd.DataFrame:
    formatted = (
        pd.concat(values, names=["algo", "version", *index_cols])
        .reset_index()
        .assign(
            algo_with_version=lambda df: (
                df["algo"] + " (" + df["version"] + ")"
            ),
            _combined="combined",
        )
    )
    return formatted


def load_raw_predictions(
    loader: ValidationResultLoader,
    algo_name: str,
    condition: str,
) -> pd.DataFrame:
    return loader.load_single_csv_file(
        algo_name, condition, "raw_predictions.csv"
    ).reset_index()


def format_loaded_predictions(
    values: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    return pd.concat(
        [
            df.assign(
                algo=algo,
                version=version,
                algo_with_version=f"{algo} ({version})",
                is_correct=lambda data: data["label"] == data["prediction"],
            )
            for (algo, version), df in values.items()
        ],
        ignore_index=True,
    )


local_data_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH")) / "results"
    if int(get_env_var("MOBGAP_VALIDATION_USE_LOCAL_DATA", 0))
    else None
)
__RESULT_VERSION = "main"
loader = ValidationResultLoader(
    "re_orientation", result_path=local_data_path, version=__RESULT_VERSION
)

free_living_index_cols = [
    "cohort",
    "participant_id",
    "time_measure",
    "recording",
    "recording_name",
    "recording_name_pretty",
]

free_living_results = format_loaded_results(
    {
        v: loader.load_single_results(k, "free_living")
        for k, v in algorithms.items()
    },
    free_living_index_cols,
)
free_living_predictions = format_loaded_predictions(
    {
        v: load_raw_predictions(loader, k, "free_living")
        for k, v in algorithms.items()
    }
)

lab_index_cols = [
    "cohort",
    "participant_id",
    "time_measure",
    "test",
    "trial",
    "test_name",
    "test_name_pretty",
]

lab_results = format_loaded_results(
    {
        v: loader.load_single_results(k, "laboratory")
        for k, v in algorithms.items()
    },
    lab_index_cols,
)
lab_predictions = format_loaded_predictions(
    {
        v: load_raw_predictions(loader, k, "laboratory")
        for k, v in algorithms.items()
    }
)

cohort_order = ["HA", "CHF", "COPD", "MS", "PD", "PFF"]

# %%
# Performance Metrics
# -------------------
# We report two accuracy views:
#
# - The average per-recording accuracy. Every datapoint contributes one value.
# - The combined accuracy over every simulated walking-bout orientation row.
#
# The confusion matrices use the combined predictions.
from functools import partial

from mobgap.pipeline.evaluation import CustomErrorAggregations as A
from mobgap.utils.df_operations import (
    CustomOperation,
    apply_aggregations,
    apply_transformations,
)
from mobgap.utils.tables import FormatTransformer as F
from mobgap.utils.tables import RevalidationInfo, revalidation_table_styles

custom_aggs = [
    CustomOperation(
        identifier=None,
        function=A.n_datapoints,
        column_name=[("n_datapoints", "all")],
    ),
    ("accuracy", ["mean", A.conf_intervals]),
]

format_transforms = [
    CustomOperation(
        identifier=None,
        function=lambda df_: df_[("n_datapoints", "all")].astype(int),
        column_name="n_datapoints",
    ),
    CustomOperation(
        identifier=None,
        function=partial(
            F.value_with_metadata,
            value_col=("mean", "accuracy"),
            other_columns={"range": ("conf_intervals", "accuracy")},
        ),
        column_name="accuracy",
    ),
]

def validation_thresholds(datapoint_label: str) -> dict[str, RevalidationInfo]:
    return {
        f"Accuracy per {datapoint_label}": RevalidationInfo(
            threshold=0.8, higher_is_better=True
        ),
        "Combined accuracy": RevalidationInfo(threshold=0.8, higher_is_better=True),
    }


def calculate_combined_accuracy(
    predictions: pd.DataFrame,
    groupby: list[str],
) -> pd.Series:
    return predictions.groupby(groupby)["is_correct"].mean().rename(
        "combined_accuracy"
    )


def format_tables(
    single_results: pd.DataFrame,
    raw_predictions: pd.DataFrame,
    groupby: list[str],
    datapoint_label: str,
) -> pd.DataFrame:
    final_names = {
        "n_datapoints": f"# {datapoint_label}s",
        "accuracy": f"Accuracy per {datapoint_label}",
        "combined_accuracy": "Combined accuracy",
    }
    formatted_single_results = (
        single_results.groupby(groupby)
        .apply(apply_aggregations, custom_aggs, include_groups=False)
        .pipe(apply_transformations, format_transforms)
        .rename(columns=final_names)
        .loc[:, [final_names["n_datapoints"], final_names["accuracy"]]]
    )
    combined_accuracy = calculate_combined_accuracy(raw_predictions, groupby)
    formatted_single_results["Combined accuracy"] = combined_accuracy
    return formatted_single_results.loc[:, list(final_names.values())]


def calculate_confusion_matrix(predictions: pd.DataFrame) -> pd.DataFrame:
    known_labels = list(REORIENTATION_LABELS)
    extra_labels = sorted(
        set(predictions["label"]).union(predictions["prediction"])
        - set(known_labels)
    )
    labels = [*known_labels, *extra_labels]
    matrix = pd.crosstab(predictions["label"], predictions["prediction"])
    return matrix.reindex(index=labels, columns=labels, fill_value=0)


# %%
# Free-Living Comparison
# ----------------------
# The free-living condition is the expected use case for unknown sensor mounting
# orientations.
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()
sns.boxplot(
    data=free_living_results,
    x="algo_with_version",
    y="accuracy",
    ax=ax,
)
plt.xticks(rotation=45, ha="right")
fig.tight_layout()
fig.show()

free_living_perf_metrics_all = format_tables(
    free_living_results,
    free_living_predictions,
    ["algo", "version"],
    "recording",
)
free_living_perf_metrics_all.style.pipe(
    revalidation_table_styles, validation_thresholds("recording"), ["algo"]
)

# %%
# Per Cohort
# ~~~~~~~~~~
fig, ax = plt.subplots()
sns.boxplot(
    data=free_living_results,
    x="cohort",
    y="accuracy",
    hue="algo_with_version",
    order=cohort_order,
    ax=ax,
)
ax.set_title("Free-living accuracy per recording")
fig.show()

free_living_perf_metrics_cohort = (
    format_tables(
        free_living_results,
        free_living_predictions,
        ["cohort", "algo", "version"],
        "recording",
    )
    .loc[cohort_order]
)
free_living_perf_metrics_cohort.style.pipe(
    revalidation_table_styles, validation_thresholds("recording"), ["cohort", "algo"]
)

# %%
# Confusion Matrices
# ~~~~~~~~~~~~~~~~~~
fig, axes = plt.subplots(
    1,
    len(algorithms),
    figsize=(6 * len(algorithms), 5),
    constrained_layout=True,
)
if len(algorithms) == 1:
    axes = [axes]

for ax, ((algo, version), data) in zip(
    axes, free_living_predictions.groupby(["algo", "version"], sort=False)
):
    sns.heatmap(
        calculate_confusion_matrix(data),
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
    )
    ax.set_title(f"{algo} ({version})")

fig.suptitle("Free-living confusion matrices")
fig.show()

# %%
# Laboratory Comparison
# ---------------------
# Every datapoint below is one lab trial. The simulated orientation rows are
# created within every reference walking bout of that trial.
fig, ax = plt.subplots()
sns.boxplot(data=lab_results, x="algo_with_version", y="accuracy", ax=ax)
plt.xticks(rotation=45, ha="right")
fig.tight_layout()
fig.show()

lab_perf_metrics_all = format_tables(
    lab_results, lab_predictions, ["algo", "version"], "trial"
)
lab_perf_metrics_all.style.pipe(
    revalidation_table_styles, validation_thresholds("trial"), ["algo"]
)

# %%
# Per Cohort
# ~~~~~~~~~~
fig, ax = plt.subplots()
sns.boxplot(
    data=lab_results,
    x="cohort",
    y="accuracy",
    hue="algo_with_version",
    order=cohort_order,
    ax=ax,
)
ax.set_title("Laboratory accuracy per trial")
fig.show()

lab_perf_metrics_cohort = (
    format_tables(
        lab_results,
        lab_predictions,
        ["cohort", "algo", "version"],
        "trial",
    )
    .loc[cohort_order]
)
lab_perf_metrics_cohort.style.pipe(
    revalidation_table_styles, validation_thresholds("trial"), ["cohort", "algo"]
)

# %%
# Confusion Matrices
# ~~~~~~~~~~~~~~~~~~
fig, axes = plt.subplots(
    1,
    len(algorithms),
    figsize=(6 * len(algorithms), 5),
    constrained_layout=True,
)
if len(algorithms) == 1:
    axes = [axes]

for ax, ((algo, version), data) in zip(
    axes, lab_predictions.groupby(["algo", "version"], sort=False)
):
    sns.heatmap(
        calculate_confusion_matrix(data),
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
    )
    ax.set_title(f"{algo} ({version})")

fig.suptitle("Laboratory confusion matrices")
fig.show()


# sphinx_gallery_multi_image = "single"
