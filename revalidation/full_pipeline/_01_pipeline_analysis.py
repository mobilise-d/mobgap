"""
.. _pipeline_val_results:

Performance of the Mobilise-D algorithm pipeline on the TVS dataset
==============================================================

.. warning:: On this page you will find preliminary results for a standardized revalidation of the pipeline and all
  of its algorithm.
  The current state, **TECHNICAL EXPERIMENTATION**.
  Don't use these results or make any assumptions based on them.
  We will update this page incrementally and provide further information, as soon as the state of any of the validation
  steps changes.

The following provides an analysis and comparison of the Mobilise-D algorithm pipeline on the TVS dataset
for the estimation of cadence, stride length, and walking speed (lab and free-living).
We look into the actual performance of the algorithms compared to the reference data.

.. note:: If you are interested in how these results are calculated, head over to the
    :ref:`processing page <pipeline_val_gen>`.

"""

# %%
# Below are the list of algorithms that we will compare.
# Note, that we use the prefix "new" to refer to the reimplemented python algorithms.

algorithms = {
    "Official_MobiliseD_Pipeline": ("MobiliseD_Pipeline", "new"),
    "EScience_MobiliseD_Pipeline": ("MobiliseD_Pipeline", "old"),
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
from mobgap.data.validation_results import ValidationResultLoader
from mobgap.utils.misc import get_env_var


def format_loaded_results(
    values: dict[tuple[str, str], pd.DataFrame],
    index_cols: list[str],
    col_prefix_filter: str,
    convert_rel_error: bool = False,
) -> pd.DataFrame:
    formatted = (
        pd.concat(values, names=["algo", "version", *index_cols])
        .filter(like=col_prefix_filter)
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
    formatted.columns = formatted.columns.str.removeprefix(col_prefix_filter)
    return formatted


local_data_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH")) / "results"
    if int(get_env_var("MOBGAP_VALIDATION_USE_LOCAL_DATA", 1))
    else None
)
loader = ValidationResultLoader(
    "full_pipeline", result_path=local_data_path, version="full_pipeline_reval"
)

# Loading free-living data
free_living_index_cols = [
    "cohort",
    "participant_id",
    "time_measure",
    "recording",
    "recording_name",
    "recording_name_pretty",
]

_free_living_results = {
    v: loader.load_single_results(k, "free_living")
    for k, v in algorithms.items()
}
free_living_results_combined = format_loaded_results(
    _free_living_results,
    free_living_index_cols,
    "combined__",
    convert_rel_error=True,
)
free_living_results_matched = format_loaded_results(
    _free_living_results,
    free_living_index_cols,
    "matched__",
    convert_rel_error=True,
)

del _free_living_results

# Loading laboratory data
laboratory_index_cols = [
    "cohort",
    "participant_id",
    "time_measure",
    "test",
    "trial",
    "test_name",
    "test_name_pretty",
]
_laboratory_results = {
    v: loader.load_single_results(k, "laboratory")
    for k, v in algorithms.items()
}
laboratory_results_combined = format_loaded_results(
    _laboratory_results,
    laboratory_index_cols,
    "combined__",
    convert_rel_error=True,
)
laboratory_results_matched = format_loaded_results(
    _laboratory_results,
    laboratory_index_cols,
    "matched__",
    convert_rel_error=True,
)
del _laboratory_results
cohort_order = ["HA", "CHF", "COPD", "MS", "PD", "PFF"]

# %%
# Performance metrics
# -------------------
# Below you can find the setup for all performance metrics that we will calculate.
# We only use the `single__` results for the comparison.
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
        function=lambda df_: df_["walking_speed_mps__detected"].isna().sum(),
        column_name=[("n_nan_detected", "all")],
    ),
    ("walking_speed_mps__detected", ["mean", A.conf_intervals]),
    ("walking_speed_mps__reference", ["mean", A.conf_intervals]),
    ("walking_speed_mps__error", ["mean", A.loa]),
    ("walking_speed_mps__abs_error", ["mean", A.conf_intervals]),
    ("walking_speed_mps__rel_error", ["mean", A.conf_intervals]),
    ("walking_speed_mps__abs_rel_error", ["mean", A.conf_intervals]),
    CustomOperation(
        identifier=None,
        function=partial(
            A.icc,
            reference_col_name="walking_speed_mps__reference",
            detected_col_name="walking_speed_mps__detected",
            icc_type="icc2",
            # For the lab data, some trials have no results for the old algorithms.
            nan_policy="omit",
        ),
        column_name=[("icc", "trial_level"), ("icc_ci", "trial_level")],
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
            "walking_speed_mps__reference",
            "walking_speed_mps__detected",
            "walking_speed_mps__abs_error",
            "walking_speed_mps__rel_error",
            "walking_speed_mps__abs_rel_error",
        ]
    ),
    CustomOperation(
        identifier=None,
        function=partial(
            F.value_with_range,
            value_col=("mean", "walking_speed_mps__error"),
            range_col=("loa", "walking_speed_mps__error"),
        ),
        column_name="walking_speed_mps__error",
    ),
    CustomOperation(
        identifier=None,
        function=partial(
            F.value_with_range,
            value_col=("icc", "trial_level"),
            range_col=("icc_ci", "trial_level"),
        ),
        column_name="icc",
    ),
]


final_names = {
    "n_datapoints": "# participants",
    "walking_speed_mps__detected": "WD mean and CI [m/s]",
    "walking_speed_mps__reference": "INDIP mean and CI [m/s]",
    "walking_speed_mps__error": "Bias and LoA [m/s]",
    "walking_speed_mps__abs_error": "Abs. Error [m/s]",
    "walking_speed_mps__rel_error": "Rel. Error [%]",
    "walking_speed_mps__abs_rel_error": "Abs. Rel. Error [%]",
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
# Combined/Aggregated Analysis
# ****************************
# To mimic actual use of wearable device where reference data may not be available, we performed a second evaluation
# for which we combined all WBs for a Laboratory test and 2.5 h recording in the real world by taking the median of the
# calculated DMOs. These combined values were then compared between the systems.
# ****************************
# All results across all cohorts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The results below represent the average performance across all participants independent of the
# cohort in terms of absolute relative error.
import matplotlib.pyplot as plt
import seaborn as sns

metric = "abs_rel_error"  # For filtering results
metric_pretty = "Abs. Rel. Error (%)"  # For plotting
overall_df = (
    free_living_results_combined[
        [
            f"walking_speed_mps__{metric}",
            f"stride_length_m__{metric}",
            f"cadence_spm__{metric}",
        ]
    ]
    .rename(
        columns={
            f"walking_speed_mps__{metric}": "walking_speed_mps",
            f"stride_length_m__{metric}": "stride_length_m",
            f"cadence_spm__{metric}": "cadence_spm",
        }
    )
    .melt(var_name="Metric", value_name=metric_pretty)
)

fig, ax = plt.subplots()
sns.boxplot(data=overall_df, x="Metric", y=metric_pretty, ax=ax)
fig.show()

combined_perf_metrics_all = (
    free_living_results_combined.groupby(["algo", "version"])
    .apply(apply_aggregations, custom_aggs, include_groups=False)
    .pipe(format_tables)
)
combined_perf_metrics_all.style.pipe(
    revalidation_table_styles, validation_thresholds, ["algo"]
)
# %% Residual plot
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_residuals(df, stats, version, title):
    """Generates a residual plot for a given version of the algorithm ('new' or 'old')."""
    df_filtered = df[df['version'] == version].dropna()

    # Compute statistics
    mean_error, (LoA_lower, LoA_upper) = stats.value, stats.err_range

    # Regression Analysis
    slope, intercept, r_value, p_value, _ = linregress(
        df_filtered['walking_speed_mps__reference'], df_filtered['walking_speed_mps__error']
    )
    x_vals = np.linspace(df_filtered['walking_speed_mps__reference'].min(),
                         df_filtered['walking_speed_mps__reference'].max(), 100)
    y_vals = slope * x_vals + intercept

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(data=df_filtered, x='walking_speed_mps__reference', y='walking_speed_mps__error',
                              hue='cohort', palette='tab10', s=100, edgecolor='black')

    # Mean and LoA lines
    for y_val, label, style in zip([mean_error, LoA_upper, LoA_lower], ["Mean", "Upper LoA", "Lower LoA"],
                                   ['-', ':', ':']):
        plt.axhline(y_val, color='black', linestyle=style, linewidth=2 if style == '-' else 1.5)
        plt.text(1.5, y_val + 0.02, label, fontsize=12, color="black", ha="left")
        plt.text(1.5, y_val - 0.08, f"{y_val:.2f}", fontsize=12, color="black", ha="left")

    # Regression line
    plt.plot(x_vals, y_vals, 'k--', linewidth=4)

    # Regression box
    plt.text(0.1, 0.6, f"Regression:\nR = {r_value:.2f}\nP-value = {p_value:.3f}", fontsize=12, color="black",
             ha="left",
             bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"))

    # Customize legend (horizontal)
    scatter.legend_.remove()
    handles, labels = scatter.get_legend_handles_labels()
    plt.legend(handles, labels, title="Cohort", loc="lower center", fontsize=10, frameon=True,
               ncol=3)

    # Axis settings
    plt.xlim(0, 2)
    plt.ylim(-0.8, 0.8)
    plt.xlabel('Reference (m/s)')
    plt.ylabel('Error (m/s)')
    plt.title(f'Walking speed - {title}')
    plt.show()


# Generate residual plots for 'new' and 'old' data
version = "new"
stats = combined_perf_metrics_all.loc[('MobiliseD_Pipeline', version)][final_names["walking_speed_mps__error"]]
plot_residuals(free_living_results_combined, stats, version, 'New')
version = "old"
stats = combined_perf_metrics_all.loc[('MobiliseD_Pipeline', version)][final_names["walking_speed_mps__error"]]
plot_residuals(free_living_results_combined, stats,version, 'Old')

# %%
# Per Cohort
# ~~~~~~~~~~
# The results below represent the average performance across all participants within a cohort.
fig, ax = plt.subplots()
sns.boxplot(
    data=free_living_results_combined,
    x="cohort",
    y="walking_speed_mps__abs_error",
    hue="algo_with_version",
    order=cohort_order,
    ax=ax,
)
fig.show()
perf_metrics_cohort = (
    free_living_results_combined.groupby(["cohort", "algo", "version"])
    .apply(apply_aggregations, custom_aggs, include_groups=False)
    .pipe(format_tables)
    .loc[cohort_order]
)
perf_metrics_cohort.style.pipe(
    revalidation_table_styles, validation_thresholds, ["cohort", "algo"]
)
perf_metrics_cohort.to_excel(r"C:\Users\paolo\Downloads\cohort_combined_rw.xlsx")

# %%
# Deep Dive Analysis of Main Algorithms
# -------------------------------------
# Below, you can find detailed correlation and residual plots comparing the new and the old implementation of each
# algorithm.
# Each datapoint represents one participant.

from mobgap.plotting import (
    calc_min_max_with_margin,
    make_square,
    plot_regline,
    residual_plot,
)


def combo_residual_plot(data):
    fig, axs = plt.subplots(ncols=2, sharey=True, sharex=True, figsize=(15, 8))
    fig.suptitle(data.name)
    for (version, subdata), ax in zip(data.groupby("version"), axs):
        residual_plot(
            subdata,
            "walking_speed_mps__reference",
            "walking_speed_mps__detected",
            "cohort",
            "m",
            ax=ax,
            legend=ax == axs[-1],
        )
        ax.set_title(version)
    # move_legend_outside(fig, axs[-1])
    plt.tight_layout()
    plt.show()


def combo_scatter_plot(data):
    fig, axs = plt.subplots(ncols=2, sharey=True, sharex=True, figsize=(15, 8))
    fig.suptitle(data.name)
    min_max = calc_min_max_with_margin(
        data["walking_speed_mps__reference"],
        data["walking_speed_mps__detected"],
    )
    for (version, subdata), ax in zip(data.groupby("version"), axs):
        subdata = subdata[
            [
                "walking_speed_mps__reference",
                "walking_speed_mps__detected",
                "cohort",
            ]
        ].dropna(how="any")
        sns.scatterplot(
            subdata,
            x="walking_speed_mps__reference",
            y="walking_speed_mps__detected",
            hue="cohort",
            ax=ax,
            legend=ax == axs[-1],
        )
        plot_regline(
            subdata["walking_speed_mps__reference"],
            subdata["walking_speed_mps__detected"],
            ax=ax,
        )
        make_square(ax, min_max, draw_diagonal=True)
        ax.set_title(version)
        ax.set_xlabel("Reference [m]")
        ax.set_ylabel("Detected [m]")
    # move_legend_outside(fig, axs[-1])
    plt.tight_layout()
    plt.show()


free_living_results_combined.groupby("algo").apply(
    combo_residual_plot, include_groups=False
)
free_living_results_combined.groupby("algo").apply(
    combo_scatter_plot, include_groups=False
)


# %%
# Matched Analysis
# ****************
# #TODO: Explain the matched analysis
# Note, that compared to the results published in Kirk et al. (2024), the primary analysis on the matched results is
# performed on the average performance metrics across all matched WBs **per recording/per participant**.
# The original publication considered the average performance metrics across all matched WBs without additional
# aggregation.
#
# Per Cohort
# ~~~~~~~~~~
# Each datapoint in the tables and plots below is a single participant/recording.
# %%
# Per Cohort
# ~~~~~~~~~~
# The results below represent the average performance across all participants within a cohort.
fig, ax = plt.subplots()
sns.boxplot(
    data=free_living_results_matched,
    x="cohort",
    y="walking_speed_mps__abs_error",
    hue="algo_with_version",
    order=cohort_order,
    ax=ax,
)
fig.show()
perf_metrics_cohort_tpe = (
    free_living_results_matched.groupby(["cohort", "algo", "version"])
    .apply(apply_aggregations, custom_aggs, include_groups=False)
    .pipe(format_tables)
    .loc[cohort_order]
)
perf_metrics_cohort_tpe.style.pipe(
    revalidation_table_styles, validation_thresholds, ["cohort", "algo"]
)
perf_metrics_cohort_tpe.to_excel(r"C:\Users\paolo\Downloads\cohort_tpe_rw.xlsx")
# %%
# Distribution of walking speed and WB count with respect to WB duration
# -------------------------------------
matched_new_data_path = (local_data_path /
                         "full_pipeline" /
                         "free_living" /
                         "Official_MobiliseD_Pipeline" /
                         "raw_matched_errors.csv")
raw_matched_new = pd.read_csv(matched_new_data_path)

matched_old_data_path = (local_data_path /
                         "full_pipeline" /
                         "free_living" /
                         "EScience_MobiliseD_Pipeline" /
                         "raw_matched_errors.csv")
raw_matched_old = pd.read_csv(matched_old_data_path)

def plot_wb_analysis(df, title_):
    """Plots a bar diagram of WB count by duration and a boxplot of absolute walking speed error."""
    df['duration_s'] = (df['end__reference'] - df['start__reference']) / 100

    bins = {
        'All': df,
        '> 10 s': df[df['duration_s'] > 10],
        '<= 10 s': df[df['duration_s'] <= 10],
        '10 - 30 s': df[(df['duration_s'] >= 10) & (df['duration_s'] < 30)],
        '30 - 60 s': df[(df['duration_s'] >= 30) & (df['duration_s'] < 60)],
        '60 - 120 s': df[(df['duration_s'] >= 60) & (df['duration_s'] < 120)],
        '> 120 s': df[df['duration_s'] > 120]
    }

    # Bar Plot
    plt.figure(figsize=(8, 6))
    plt.bar(bins.keys(), [len(b) for b in bins.values()], color='blue')
    # Annotate each bar with its count
    counts = [len(b) for b in bins.values()]
    for i, count in enumerate(counts):
        plt.text(i, count + 5, str(count), ha='center', fontsize=12, color='black', fontweight='bold')
    plt.xlabel("Duration")
    plt.ylabel("WB Count")
    plt.title(title_)
    plt.show()
    # Box Plot
    bin_labels = []
    errors = []

    for label, b in bins.items():
        bin_labels.extend([label] * len(b))
        errors.extend(b['walking_speed_mps__abs_error'])

    df_boxplot = pd.DataFrame({'Duration': bin_labels, 'Absolute Error': errors})

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Duration', y='Absolute Error', data=df_boxplot, hue='Duration', palette='Blues', legend=False)

    plt.xlabel("Duration")
    plt.ylabel("Absolute Walking Speed Error (m/s)")
    plt.title("Distribution of Absolute Walking Speed Error by Duration")
    plt.show()

# Generate WB analysis plots
plot_wb_analysis(raw_matched_new, "New")
plot_wb_analysis(raw_matched_old, "Old")