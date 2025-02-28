"""
.. _sl_val_results:

Performance of the stride length algorithms on the TVS dataset
==============================================================

.. warning:: On this page you will find preliminary results for a standardized revalidation of the pipeline and all
  of its algorithm.
  The current state, **TECHNICAL EXPERIMENTATION**.
  Don't use these results or make any assumptions based on them.
  We will update this page incrementally and provide further information, as soon as the state of any of the validation
  steps changes.

The following provides an analysis and comparison of the stride length algorithms on the TVS dataset
(lab and free-living).
We look into the actual performance of the algorithms compared to the reference data.
Note, that at the time of writing, comparison with the original Matlab results is not possible, as these algorithms
were not run on the same version of the TVS dataset.

.. note:: If you are interested in how these results are calculated, head over to the
    :ref:`processing page <sl_val_gen>`.

"""

# %%
# Below are the list of algorithms that we will compare.
# Note, that we use the prefix "new" to refer to the reimplemented python algorithms.
# For the zjils algorithm, we compare both potential threshold values that were determined as part of the pre-validation
# analysis on the MsProject dataset.

algorithms = {
    "SlZjilstra__MS_ALL": ("SlZjilstra - MS-all", "new"),
    "SlZjilstra__MS_MS": ("SlZjilstra - MS-MS", "new"),
    "matlab_zjilsV3__MS_ALL": ("SlZjilstra - MS-all", "old"),
    "matlab_zjilsV3__MS_MS": ("SlZjilstra - MS-MS", "old"),
}

# %%
# The code below loads the data and prepares it for the analysis.
# By default, the data will be downloaded from an online repository (and cached locally).
# If you want to use a local copy of the data, you can set the `MOBGAP_VALIDATION_DATA_PATH` environment variable.
# and the MOBGAP_VALIDATION_USE_LOCA_DATA to `1`.
#
# The file download will print a couple log information, which can usually be ignored.
# You can also change the `version` parameter to load a different version of the data.
from pathlib import Path

import pandas as pd
import pingouin as pg
from mobgap.data.validation_results import ValidationResultLoader
from mobgap.utils.misc import get_env_var


def format_loaded_results(
    values: dict[tuple[str, str], pd.DataFrame],
    index_cols: list[str],
    convert_rel_error: bool = False,
) -> pd.DataFrame:
    formatted = (
        pd.concat(values, names=["algo", "version", *index_cols])
        .reset_index()
        .assign(
            algo_with_version=lambda df: df["algo"]
            + " ("
            + df["version"]
            + ")",
            _combined="combined",
        )
    )
    if not convert_rel_error:
        return formatted
    rel_cols = [c for c in formatted.columns if "rel_error" in c]
    formatted[rel_cols] = formatted[rel_cols] * 100
    return formatted


local_data_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH")) / "results"
    if int(get_env_var("MOBGAP_VALIDATION_USE_LOCAL_DATA", 0))
    else None
)
loader = ValidationResultLoader(
    "sl", result_path=local_data_path, version="main"
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
    convert_rel_error=True,
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
    convert_rel_error=True,
)

cohort_order = ["HA", "CHF", "COPD", "MS", "PD", "PFF"]
# %%
# Performance metrics
# -------------------
# Below you can find the setup for all performance metrics that we will calculate.
# We only use the `wb__` results for the comparison.
# These results are calculated by first calculating the average stride length per WB.
# Then calculating the error metrics for each WB.
# Then we take the average over all WBs of a participant to get the `wb__` results.
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
    CustomOperation(
        identifier=None,
        function=lambda df_: df_["wb__detected"].isna().sum(),
        column_name=[("n_nan_detected", "all")],
    ),
    ("wb__detected", ["mean", A.conf_intervals]),
    ("wb__reference", ["mean", A.conf_intervals]),
    ("wb__error", ["mean", A.loa]),
    ("wb__abs_error", ["mean", A.conf_intervals]),
    ("wb__rel_error", ["mean", A.conf_intervals]),
    ("wb__abs_rel_error", ["mean", A.conf_intervals]),
    CustomOperation(
        identifier=None,
        function=partial(
            A.icc,
            reference_col_name="wb__reference",
            detected_col_name="wb__detected",
            icc_type="icc2",
            # For the lab data, some trials have no results for the old algorithms.
            nan_policy="omit",
        ),
        column_name=[("icc", "wb_level"), ("icc_ci", "wb_level")],
    ),
]

format_transforms = [
    CustomOperation(
        identifier=None,
        function=lambda df_: df_[("n_datapoints", "all")].astype(int),
        column_name="n_datapoints",
    ),
    CustomOperation(
        identifier=None,
        function=lambda df_: df_[("n_nan_detected", "all")].astype(int),
        column_name="n_nan_detected",
    ),
    *(
        CustomOperation(
            identifier=None,
            function=partial(
                F.value_with_range,
                value_col=("mean", c),
                range_col=("conf_intervals", c),
            ),
            column_name=c,
        )
        for c in [
            "wb__reference",
            "wb__detected",
            "wb__abs_error",
            "wb__rel_error",
            "wb__abs_rel_error",
        ]
    ),
    CustomOperation(
        identifier=None,
        function=partial(
            F.stats_result,
            p_value_col=("T", "wb__abs_error"),
            effect_size_col=("p", "wb__abs_error"),
        ),
        column_name="wb__abs_error__stats",
    ),
    CustomOperation(
        identifier=None,
        function=partial(
            F.value_with_range,
            value_col=("mean", "wb__error"),
            range_col=("loa", "wb__error"),
        ),
        column_name="wb__error",
    ),
    CustomOperation(
        identifier=None,
        function=partial(
            F.value_with_range,
            value_col=("icc", "wb_level"),
            range_col=("icc_ci", "wb_level"),
        ),
        column_name="icc",
    ),
]


final_names = {
    "n_datapoints": "# participants",
    "wb__detected": "WD mean and CI [m]",
    "wb__reference": "INDIP mean and CI [m]",
    "wb__error": "Bias and LoA [m]",
    "wb__abs_error": "Abs. Error [m]",
    "wb__abs_error__stats": "Abs. Error Stats. [m]",
    "wb__rel_error": "Rel. Error [%]",
    "wb__abs_rel_error": "Abs. Rel. Error [%]",
    "icc": "ICC",
    "n_nan_detected": "# Failed WBs",
}

validation_thresholds = {
    "Abs. Error [m]": RevalidationInfo(threshold=None, higher_is_better=False),
    "Abs. Rel. Error [%]": RevalidationInfo(
        threshold=20, higher_is_better=False
    ),
    "ICC": RevalidationInfo(threshold=0.7, higher_is_better=True),
    "# Failed WBs": RevalidationInfo(threshold=None, higher_is_better=False),
}


def pairwise_tests(
    df: pd.DataFrame, dv: str, between: str
) -> tuple[float, float]:
    result = pg.pairwise_tests(data=df, dv=dv, between=between)
    assert len(result) == 1
    return tuple(result.iloc[0][["T", "p-unc"]].to_list())


def agg_errors(
    df: pd.DataFrame, groupby: list[str], stats_between="version"
) -> pd.DataFrame:
    error_agg = df.groupby([*groupby, stats_between]).apply(
        apply_aggregations, custom_aggs, include_groups=False
    )
    stats = df.groupby(groupby).apply(
        apply_aggregations,
        [
            CustomOperation(
                identifier=None,
                function=partial(
                    pairwise_tests, dv="wb__abs_error", between="version"
                ),
                column_name=[
                    (
                        "T",
                        "wb__abs_error",
                    ),
                    ("p", "wb__abs_error"),
                ],
            )
        ],
        include_groups=False,
    )
    return error_agg.join(stats, on=groupby, how="left")


def format_tables(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(apply_transformations, format_transforms)
        .rename(columns=final_names)
        .loc[:, list(final_names.values())]
    )


# %%
# Free-Living Comparison
# ----------------------
# We focus on the free-living data for the comparison as this is the expected use case for the algorithms.
#
# All results across all cohorts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The results below represent the average performance across all participants independent of the
# cohort.
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()
sns.boxplot(
    data=free_living_results, x="algo_with_version", y="wb__abs_error", ax=ax
)
fig.show()

perf_metrics_all = free_living_results.pipe(
    agg_errors, groupby=["algo"], stats_between="version"
).pipe(format_tables)
perf_metrics_all.style.pipe(
    revalidation_table_styles, validation_thresholds, ["algo"], stats_to="old"
)

# %%
# Per Cohort
# ~~~~~~~~~~
# The results below represent the average performance across all participants within a cohort.
fig, ax = plt.subplots()
sns.boxplot(
    data=free_living_results,
    x="cohort",
    y="wb__abs_error",
    hue="algo_with_version",
    order=cohort_order,
    ax=ax,
)
fig.show()
perf_metrics_cohort = (
    free_living_results.pipe(
        agg_errors, groupby=["cohort", "algo"], stats_between="version"
    )
    .pipe(format_tables)
    .loc[cohort_order]
)
perf_metrics_cohort.style.pipe(
    revalidation_table_styles,
    validation_thresholds,
    ["cohort", "algo"],
    stats_to="old",
)

# %%
# Deep Dive Analysis of Main Algorithms
# -------------------------------------
# Below, you can find detailed correlation and residual plots comparing the new and the old implementation of each
# algorithm.
# Each datapoint represents one participant.

from mobgap.plotting import (
    calc_min_max_with_margin,
    make_square,
    move_legend_outside,
    plot_regline,
    residual_plot,
)


def combo_residual_plot(data):
    fig, axs = plt.subplots(ncols=2, sharey=True, sharex=True, figsize=(15, 8))
    fig.suptitle(data.name)
    for (version, subdata), ax in zip(data.groupby("version"), axs):
        residual_plot(
            subdata,
            "wb__reference",
            "wb__detected",
            "cohort",
            "m",
            ax=ax,
            legend=ax == axs[-1],
        )
        ax.set_title(version)
    move_legend_outside(fig, axs[-1])
    plt.tight_layout()
    plt.show()


def combo_scatter_plot(data):
    fig, axs = plt.subplots(ncols=2, sharey=True, sharex=True, figsize=(15, 8))
    fig.suptitle(data.name)
    min_max = calc_min_max_with_margin(
        data["wb__reference"], data["wb__detected"]
    )
    for (version, subdata), ax in zip(data.groupby("version"), axs):
        subdata = subdata[["wb__reference", "wb__detected", "cohort"]].dropna(
            how="any"
        )
        sns.scatterplot(
            subdata,
            x="wb__reference",
            y="wb__detected",
            hue="cohort",
            ax=ax,
            legend=ax == axs[-1],
        )
        plot_regline(subdata["wb__reference"], subdata["wb__detected"], ax=ax)
        make_square(ax, min_max, draw_diagonal=True)
        ax.set_title(version)
        ax.set_xlabel("Reference [m]")
        ax.set_ylabel("Detected [m]")
    move_legend_outside(fig, axs[-1])
    plt.tight_layout()
    plt.show()


free_living_results.groupby("algo").apply(
    combo_residual_plot, include_groups=False
)
free_living_results.groupby("algo").apply(
    combo_scatter_plot, include_groups=False
)

# %%
# Below, we show the direct correlation between the results from the old and the new implementation.
# Each datapoint represents one participant.


def compare_scatter_plot(data):
    fig, ax = plt.subplots(figsize=(8, 8))
    reformated_data = (
        data.pivot_table(
            values="wb__detected",
            index=("cohort", "participant_id"),
            columns="version",
        )
        .reset_index()
        .dropna(how="any")
    )

    min_max = calc_min_max_with_margin(
        reformated_data["old"], reformated_data["new"]
    )
    sns.scatterplot(reformated_data, x="old", y="new", hue="cohort", ax=ax)
    plot_regline(reformated_data["old"], reformated_data["new"], ax=ax)
    make_square(ax, min_max, draw_diagonal=True)
    move_legend_outside(fig, ax)
    ax.set_title(data.name)
    ax.set_xlabel("Old algorithm version [m]")
    ax.set_ylabel("New algorithm version [m]")
    plt.tight_layout()
    plt.show()


free_living_results.groupby("algo").apply(
    compare_scatter_plot, include_groups=False
)

# %%
# Speed dependency
# ~~~~~~~~~~~~~~~~
# One important aspect of the algorithm performance is the dependency on the walking speed.
# Aka, how well do the algorithms perform at different walking speeds.
# For this we plot the absolute relative error against the walking speed of the reference data.
# For better granularity, we use the values per WB, instead of the aggregates per participant.
#
# The overlayed dots represent the trend-line calculated by taking the median of the absolute relative error within
# bins of 0.05 m/s.
import numpy as np

wb_level_results = format_loaded_results(
    {
        v: loader.load_single_csv_file(
            k, "free_living", "raw_wb_level_values_with_errors.csv"
        )
        for k, v in algorithms.items()
    },
    free_living_index_cols,
)

algo_names = wb_level_results.algo_with_version.unique()
fig, axs = plt.subplots(
    len(algo_names),
    1,
    sharex=True,
    sharey=True,
    figsize=(12, 3 * len(algo_names)),
)
for ax, algo in zip(axs, algo_names):
    data = wb_level_results.query("algo_with_version == @algo").copy()

    # Create scatter plot
    sns.scatterplot(
        data=data,
        x="reference_ws",
        y="abs_rel_error",
        ax=ax,
        alpha=0.3,
    )

    # Create bins and calculate medians
    bins = np.arange(0, data["reference_ws"].max() + 0.05, 0.05)
    data["speed_bin"] = pd.cut(data["reference_ws"], bins=bins)

    # Calculate bin centers for plotting
    data["bin_center"] = data["speed_bin"].apply(lambda x: x.mid)

    # Calculate medians per bin and cohort
    binned_data = (
        data.groupby("bin_center", observed=True)["abs_rel_error"]
        .median()
        .reset_index()
    )

    # Plot median lines
    sns.scatterplot(
        data=binned_data,
        x="bin_center",
        y="abs_rel_error",
        ax=ax,
    )

    ax.set_title(algo)
    ax.set_xlabel("Walking Speed (m/s)")
    ax.set_ylabel("Absolute Relative Error")

fig.tight_layout()
fig.show()

# %%
# Laboratory Comparison
# ----------------------
# Every datapoint below is one trial of a test.
# Note, that each datapoint is weighted equally in the calculation of the performance metrics.
# This is a limitation of this simple approach, as the number of strides per trial and the complexity of the context
# can vary significantly.
# For a full picture, different groups of tests should be analyzed separately.
# The approach below should still provide a good overview to compare the algorithms.
fig, ax = plt.subplots()
sns.boxplot(data=lab_results, x="algo_with_version", y="wb__abs_error", ax=ax)
fig.show()

perf_metrics_all = lab_results.pipe(
    agg_errors, groupby=["algo"], stats_between="version"
).pipe(format_tables)
perf_metrics_all.style.pipe(
    revalidation_table_styles, validation_thresholds, ["algo"], stats_to="old"
)

# %%
# Per Cohort
# ~~~~~~~~~~
# The results below represent the average performance across all trails of all participants within a cohort.
fig, ax = plt.subplots()
sns.boxplot(
    data=lab_results,
    x="cohort",
    y="wb__abs_error",
    hue="algo_with_version",
    order=cohort_order,
    ax=ax,
)
fig.show()
perf_metrics_cohort = (
    lab_results.pipe(
        agg_errors, groupby=["cohort", "algo"], stats_between="version"
    )
    .pipe(format_tables)
    .loc[cohort_order]
)
perf_metrics_cohort.style.pipe(
    revalidation_table_styles,
    validation_thresholds,
    ["cohort", "algo"],
    stats_to="old",
)


# sphinx_gallery_multi_image = "single"
