"""
.. _pipeline_val_results:

Performance of the Mobilise-D algorithm pipeline on the TVS dataset for walking speed estimation
================================================================================================

.. warning:: On this page you will find preliminary results for a standardized revalidation of the pipeline and all
  of its algorithm.
  The current state, **TECHNICAL EXPERIMENTATION**.
  Don't use these results or make any assumptions based on them.
  We will update this page incrementally and provide further information, as soon as the state of any of the validation
  steps changes.

The following provides an analysis and comparison of the Mobilise-D algorithm pipeline on the
`Mobilise-D Technical Validation Study (TVS) dataset <https://zenodo.org/records/13987963>`_
for the estimation of walking speed (free-living).
In this example, we look into the performance of the Python implementation of the pipeline compared to the reference
data. We also compare the actual performance to that obtained by the original Matlab-based implementation  [1]_.
.. [1] Kirk, C., Küderle, A., Micó-Amigo, M.E. et al. Mobilise-D insights to estimate real-world walking speed in
       multiple conditions with a wearable device. Sci Rep 14, 1754 (2024).
       https://doi.org/10.1038/s41598-024-51766-5

.. note:: If you are interested in how these results are calculated, head over to the
    :ref:`processing page <pipeline_val_gen>`.

"""

# %%
# Below the list of pipelines that are compared is shown.
# Note, that we use the prefix "new" to refer to the reimplemented python algorithms, and the prefix "old" to refer to
# the original Matlab-based implementation.

algorithms = {
    "Official_MobiliseD_Pipeline": ("MobiliseD_Pipeline", "new"),
    "EScience_MobiliseD_Pipeline": ("MobiliseD_Pipeline", "old"),
}
# %%
# The code below loads the data and prepares it for the analysis.
# By default, the data will be downloaded from an online repository (and cached locally).
# If you want to use a local copy of the data, you can set the `MOBGAP_VALIDATION_DATA_PATH` environment variable.
# and the `MOBGAP_VALIDATION_USE_LOCA_DATA` to `1`.
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
    use_col_prefix_filter: bool = True,  # Single optional argument
) -> pd.DataFrame:
    formatted = (
        pd.concat(values, names=["algo", "version", *index_cols])
        # Apply filtering only if use_col_prefix_filter is True
        .pipe(lambda df: df.filter(like=col_prefix_filter) if use_col_prefix_filter else df)
        .reset_index()
        .assign(
            algo_with_version=lambda df: df["algo"] + " (" + df["version"] + ")",
            _combined="combined",
        )
    )

    # If use_col_prefix_filter is True, apply filtering and remove prefix
    if use_col_prefix_filter:
        formatted.columns = formatted.columns.str.removeprefix(col_prefix_filter)

    if convert_rel_error:
        rel_cols = [c for c in formatted.columns if "rel_error" in c]
        formatted[rel_cols] = formatted[rel_cols] * 100

    return formatted

local_data_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH")) / "results"
    if int(get_env_var("MOBGAP_VALIDATION_USE_LOCAL_DATA", 0))
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

_free_living_results = { # Matched and aggregate/combined per-recording results for the 2.5 h free-living recordings
    v: loader.load_single_results(k, "free_living")
    for k, v in algorithms.items()
}

_free_living_results_raw = { # Matched per-WB results for the 2.5 h free-living recordings
    v: loader.load_single_csv_file(k, "free_living", "raw_matched_errors.csv")
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
free_living_results_matched_raw = format_loaded_results(
    values = _free_living_results_raw,
    index_cols = free_living_index_cols,
    col_prefix_filter = "matched__", # This argument is actually not used
    use_col_prefix_filter = False,
    convert_rel_error=True,
)

del _free_living_results, _free_living_results_raw
cohort_order = ["HA", "CHF", "COPD", "MS", "PD", "PFF"]
# %%

# Performance metrics
# -------------------
# Below you can find the setup for all performance metrics that we will calculate.
# We only use the `single__` results for the comparison.
# These results are calculated by first calculating the average walking speed per walking bout (WB).
# Then all WBs of a participant are aggregated. Eventually, the error metrics for each WB are calculated.
#
# .. note:: For the evaluation of the full pipeline performance, two types of aggregation are performed, which will be
#           described later on in the example.
#
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
    "Abs. Error [m/s]": RevalidationInfo(threshold=None, higher_is_better=False),
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
# Combined/Aggregated Evaluation
# ------------------------------------------
# To mimic actual use of wearable device where reference data may not be available, we performed a first evaluation
# for which we combined all WBs for a datapoint by taking the median of the calculated walking speed. These combined
# values were then compared between the systems.
#
# .. note:: In the free-living dataset, each datapoint represents one 2.5h recording. In the laboratory dataset, each
#           datapoint represents one trial.
#
# All results across all cohorts
# ******************************
# The results below represent the average performance across all participants independent of the
# cohort in terms of error, relative error, absolute error, and absolute relative error.

import matplotlib.pyplot as plt
import seaborn as sns

fontsize_ = 20
# Define the four metrics to plot (for walking speed only)
metrics = ["abs_rel_error", "error", "rel_error", "abs_error"]
metric_titles = {
    "abs_rel_error": "Abs. Rel. Error (%)",
    "error": "Error (m/s)",
    "rel_error": "Rel. Error (%)",
    "abs_error": "Abs. Error (m/s)",
}

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()  # Flatten for easier iteration

# Loop through metrics and plot each in a separate subplot
for i, metric in enumerate(metrics):
    metric_pretty = metric_titles[metric]

    # Prepare DataFrame for boxplot (only walking speed)
    overall_df = (
        free_living_results_combined[
            ["version", f"walking_speed_mps__{metric}"]
        ]
        .rename(columns={f"walking_speed_mps__{metric}": metric_pretty})
    )

    # Create paired boxplots by version (old vs new)
    sns.boxplot(data=overall_df, x="version", hue="version", y=metric_pretty, ax=axes[i])

    # Title and labels with increased fontsize
    axes[i].set_title(metric_pretty, fontsize=fontsize_)
    axes[i].set_ylabel(metric_pretty, fontsize=fontsize_)

    # Set fontsize for tick labels
    axes[i].tick_params(axis='both', which='major', labelsize=fontsize_)
    axes[i].tick_params(axis='both', which='minor', labelsize=fontsize_)

    # Add grid to the plot
    axes[i].grid(True)  # Add grid lines to the plot

# Improve layout and show plot
plt.tight_layout()
plt.show()
# %%
# Processing the performance table
combined_perf_metrics_all = (
    free_living_results_combined.groupby(["algo", "version"])
    .apply(apply_aggregations, custom_aggs, include_groups=False)
    .pipe(format_tables)
)

# Apply styling to the table
combined_perf_metrics_all.style.pipe(
    revalidation_table_styles, validation_thresholds, ["algo"]
)
# %% Residual plot
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_residuals(df, stats, version, title, ax):
    """Generates a residual plot for a given version of the algorithm ('new' or 'old') on the provided axis."""
    if 'version' in df.columns:
        df_filtered = df[df['version'] == version].dropna()
    else:
        df_filtered = df.dropna()

    # Compute statistics
    mean_error, (LoA_lower, LoA_upper) = stats.value, stats.err_range

    # Regression Analysis
    slope, intercept, r_value, p_value, _ = linregress(
        df_filtered['walking_speed_mps__reference'], df_filtered['walking_speed_mps__error']
    )
    x_vals = np.linspace(df_filtered['walking_speed_mps__reference'].min(),
                         df_filtered['walking_speed_mps__reference'].max(), 100)
    y_vals = slope * x_vals + intercept

    # Scatter plot
    scatter = sns.scatterplot(data=df_filtered, x='walking_speed_mps__reference', y='walking_speed_mps__error',
                              hue='cohort', palette='tab10', s=100, edgecolor='black', ax=ax)

    # Mean and LoA lines
    for y_val, label, style in zip([mean_error, LoA_upper, LoA_lower], ["Mean", "Upper LoA", "Lower LoA"],
                                   ['-', ':', ':']):
        ax.axhline(y_val, color='black', linestyle=style, linewidth=2 if style == '-' else 1.7)
        ax.text(1.7, y_val + 0.02, label, fontsize=12, color="black", ha="left")
        ax.text(1.7, y_val - 0.08, f"{y_val:.2f}", fontsize=12, color="black", ha="left")

    # Regression line
    ax.plot(x_vals, y_vals, 'k--', linewidth=4)

    # Regression box
    ax.text(0.1, 0.6, f"Regression:\nR = {r_value:.2f}\nP-value = {p_value:.3f}", fontsize=12, color="black",
             ha="left",
             bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"))

    # Customize legend (horizontal)
    scatter.legend_.remove()
    handles, labels = scatter.get_legend_handles_labels()
    ax.legend(handles, labels, title="Cohort", loc="lower center", fontsize=10, frameon=True,
              ncol=3)

    # Axis settings
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.8, 0.8)
    ax.set_xlabel('Reference (m/s)')
    ax.set_ylabel('Error (m/s)')
    ax.set_title(f'Walking speed - {title}')


# Create a 1x2 subplot figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

# Generate residual plot for the 'new' version
version = "new"
stats = combined_perf_metrics_all.loc[('MobiliseD_Pipeline', version)][final_names["walking_speed_mps__error"]]
plot_residuals(free_living_results_combined, stats, version, 'New', axes[0])

# Generate residual plot for the 'old' version
version = "old"
stats = combined_perf_metrics_all.loc[('MobiliseD_Pipeline', version)][final_names["walking_speed_mps__error"]]
plot_residuals(free_living_results_combined, stats, version, 'Old', axes[1])

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
# %%
# Per Cohort
# ~~~~~~~~~~
# The results below represent the average absolute error on walking speed estimation
# across all participants within a cohort.
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
# %%
combined_perf_metrics_cohort = (
    free_living_results_combined.groupby(["cohort", "algo", "version"])
    .apply(apply_aggregations, custom_aggs, include_groups=False)
    .pipe(format_tables)
    .loc[cohort_order]
)
combined_perf_metrics_cohort.style.pipe(
    revalidation_table_styles, validation_thresholds, ["cohort", "algo"]
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
# Matched/True Positive Evaluation
# ****************
# The "Matched" Evaluation directly compares the performance of walking speed estimation on only the WBs that were
# detected in both systems (true positives). This allows for the calculation of traditional comparison metrics
# (e.g., interclass correlation and Bland–Altman plots), that require a direct comparison of individual measurement
# points. WBs were included in the true positive analysis, if there was an overlap of more than 80% between WBs detected
# by the two systems (details about the selection of this threshold can be found in Kirk et al. (2024)). The threshold
# of 80% was selected as a trade-off to allow us: (i) to consider as much as possible a like-for-like comparison between
# selected WBs (INDIP vs. wearable device), and at the same time (ii) to include the minimum number of WBs to ensure
# sufficient statistical power for the analyses (i.e., at least 101 walking bouts for each cohort). This target was
# based upon the number of WBs rather than a percentage of total walking bouts that would allow us to meet criteria
# established by statistical experts for robust statistical analysis after sample-size re-evaluation
# (total WB number > 101 corresponding to ICC > 0.7 and a CI = 0.2).
# Note, that compared to the results published in Kirk et al. (2024), the primary analysis on the matched results is
# performed on the average performance metrics across all matched WBs **per recording/per participant**.
# The original publication considered the average performance metrics across all matched WBs without additional
# aggregation.
#
# ****************************
# All results across all cohorts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The results below represent the average performance across all participants independent of the
# cohort in terms of error, relative error, absolute error, and absolute relative error.

import matplotlib.pyplot as plt
import seaborn as sns

fontsize_ = 20
# Define the four metrics to plot (for walking speed only)
metrics = ["abs_rel_error", "error", "rel_error", "abs_error"]
metric_titles = {
    "abs_rel_error": "Abs. Rel. Error (%)",
    "error": "Error (m/s)",
    "rel_error": "Rel. Error (%)",
    "abs_error": "Abs. Error (m/s)",
}

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()  # Flatten for easier iteration

# Loop through metrics and plot each in a separate subplot
for i, metric in enumerate(metrics):
    metric_pretty = metric_titles[metric]

    # Prepare DataFrame for boxplot (only walking speed)
    overall_df = (
        free_living_results_matched[
            ["version", f"walking_speed_mps__{metric}"]
        ]
        .rename(columns={f"walking_speed_mps__{metric}": metric_pretty})
    )

    # Create paired boxplots by version (old vs new)
    sns.boxplot(data=overall_df, x="version", hue="version", y=metric_pretty, ax=axes[i])

    # Title and labels with increased fontsize
    axes[i].set_title(metric_pretty, fontsize=fontsize_)
    axes[i].set_ylabel(metric_pretty, fontsize=fontsize_)

    # Set fontsize for tick labels
    axes[i].tick_params(axis='both', which='major', labelsize=fontsize_)
    axes[i].tick_params(axis='both', which='minor', labelsize=fontsize_)

    # Add grid to the plot
    axes[i].grid(True)  # Add grid lines to the plot

# Improve layout and show plot
plt.tight_layout()
plt.show()
# %%
# Processing the performance table
matched_perf_metrics_all = (
    free_living_results_matched.groupby(["algo", "version"])
    .apply(apply_aggregations, custom_aggs, include_groups=False)
    .pipe(format_tables)
)

# Apply styling to the table
matched_perf_metrics_all.style.pipe(
    revalidation_table_styles, validation_thresholds, ["algo"]
)
# %%
# Per Cohort
# ~~~~~~~~~~
# Each datapoint in the tables and plots below is a single participant/recording.
# The results below represent the average absolute error across all participants within a cohort.
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
# %%
matched_perf_metrics_cohort = (
    free_living_results_matched.groupby(["cohort", "algo", "version"])
    .apply(apply_aggregations, custom_aggs, include_groups=False)
    .pipe(format_tables)
    .loc[cohort_order]
)
matched_perf_metrics_cohort.style.pipe(
    revalidation_table_styles, validation_thresholds, ["cohort", "algo"]
)

matched_perf_metrics_all = (
    free_living_results_matched.groupby(["algo", "version"])
    .apply(apply_aggregations, custom_aggs, include_groups=False)
    .pipe(format_tables)
)
# %% Residual plot
# Create a 1x2 subplot figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

# Generate residual plot for the 'new' version
version = "new"
stats = matched_perf_metrics_all.loc[('MobiliseD_Pipeline', version)][final_names["walking_speed_mps__error"]]
plot_residuals(free_living_results_matched_raw, stats, version, 'New', axes[0])

# Generate residual plot for the 'old' version
version = "old"
stats = matched_perf_metrics_all.loc[('MobiliseD_Pipeline', version)][final_names["walking_speed_mps__error"]]
plot_residuals(free_living_results_matched_raw, stats, version, 'Old', axes[1])

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
# %%
# We investigate the dependency of the absolute walking speed error of all true-positive WBs from the real-world
# recording on the WB duration reported by the reference system. In the top, WB errors are grouped by various duration
# bouts. In the bottom the number of bouts within each duration group is visualized.
def plot_wb_analysis_combined(df):
    """Generates a single figure with:
       - First row: Two side-by-side boxplot for "new" and "old" cases.
       - Second row: A grouped bar chart comparing WB counts for "new" and "old" cases.

       df: DataFrame containing 'version' column with values 'new' or 'old' to distinguish data
    """
    fontsize_ = 20  # Set font size for all plot elements

    # Compute WB durations in seconds
    df['duration_s'] = (df['end__reference'] - df['start__reference']) / 100

    # Define WB duration bins
    bin_labels = ['All', '> 10 s', '<= 10 s', '10 - 30 s', '30 - 60 s', '60 - 120 s', '> 120 s']
    bins_new = {
        'All': df[df['version'] == 'new'],
        '> 10 s': df[(df['version'] == 'new') & (df['duration_s'] > 10)],
        '<= 10 s': df[(df['version'] == 'new') & (df['duration_s'] <= 10)],
        '10 - 30 s': df[(df['version'] == 'new') & (df['duration_s'] >= 10) & (df['duration_s'] < 30)],
        '30 - 60 s': df[(df['version'] == 'new') & (df['duration_s'] >= 30) & (df['duration_s'] < 60)],
        '60 - 120 s': df[(df['version'] == 'new') & (df['duration_s'] >= 60) & (df['duration_s'] < 120)],
        '> 120 s': df[(df['version'] == 'new') & (df['duration_s'] > 120)]
    }
    bins_old = {
        'All': df[df['version'] == 'old'],
        '> 10 s': df[(df['version'] == 'old') & (df['duration_s'] > 10)],
        '<= 10 s': df[(df['version'] == 'old') & (df['duration_s'] <= 10)],
        '10 - 30 s': df[(df['version'] == 'old') & (df['duration_s'] >= 10) & (df['duration_s'] < 30)],
        '30 - 60 s': df[(df['version'] == 'old') & (df['duration_s'] >= 30) & (df['duration_s'] < 60)],
        '60 - 120 s': df[(df['version'] == 'old') & (df['duration_s'] >= 60) & (df['duration_s'] < 120)],
        '> 120 s': df[(df['version'] == 'old') & (df['duration_s'] > 120)]
    }

    # Prepare boxplot data (combine "new" and "old" in a single DataFrame)
    boxplot_data = []
    for label in bin_labels:
        boxplot_data.extend([(label, 'New', val) for val in bins_new[label]['walking_speed_mps__abs_error']])
        boxplot_data.extend([(label, 'Old', val) for val in bins_old[label]['walking_speed_mps__abs_error']])

    df_boxplot = pd.DataFrame(boxplot_data, columns=['Duration', 'Version', 'Absolute Error'])

    # Prepare bar plot data
    bar_counts_new = [len(bins_new[label]) for label in bin_labels]
    bar_counts_old = [len(bins_old[label]) for label in bin_labels]

    x = np.arange(len(bin_labels))  # X locations for the bars
    bar_width = 0.4  # Width of the bars

    # Create the figure with subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

    # --- Boxplot (First Row) ---
    sns.boxplot(x='Duration', y='Absolute Error', hue='Version', data=df_boxplot, palette='Blues', ax=axes[0])
    axes[0].set_xlabel("WB Duration", fontsize=fontsize_)
    axes[0].set_ylabel("Absolute Walking Speed Error (m/s)", fontsize=fontsize_)
    axes[0].set_title("Distribution of Absolute Walking Speed Error by Duration", fontsize=fontsize_)
    axes[0].legend(title="Version", loc="upper right", fontsize=fontsize_)

    # --- Bar Plot (Second Row) ---
    axes[1].bar(x - bar_width / 2, bar_counts_new, bar_width, label='New', color='blue', alpha=0.7)
    axes[1].bar(x + bar_width / 2, bar_counts_old, bar_width, label='Old', color='orange', alpha=0.7)

    # Annotate bar counts
    for i in range(len(bin_labels)):
        axes[1].text(x[i] - bar_width / 2, bar_counts_new[i] + 5, str(bar_counts_new[i]),
                     ha='center', fontsize=fontsize_, color='black', fontweight='bold')
        axes[1].text(x[i] + bar_width / 2, bar_counts_old[i] + 5, str(bar_counts_old[i]),
                     ha='center', fontsize=fontsize_, color='black', fontweight='bold')

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bin_labels, fontsize=fontsize_)
    axes[1].set_xlabel("WB Duration", fontsize=fontsize_)
    axes[1].set_ylabel("WB Count", fontsize=fontsize_)
    axes[1].set_title("Comparison of WB Count by Duration", fontsize=fontsize_)
    axes[1].legend(title="Version", loc="upper right", fontsize=fontsize_)

    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()


# Call the function with the combined data (new and old) including the 'version' column
plot_wb_analysis_combined(free_living_results_matched_raw)
# %% Speed dependency
# One important aspect of the algorithm performance is the dependency on the walking speed. Aka, how well do the
# algorithms perform at different walking speeds. For this we plot the absolute relative error against the walking speed
# of the reference data. For better granularity, we use the values per WB, instead of the aggregates per participant.
# The overlayed dots represent the trend-line calculated by taking the median of the absolute relative error within bins
# of 0.05 m/s.

# For plotting all participants at the end
combined = free_living_results_matched_raw.copy()
combined["cohort"] = "Combined"
ws_level_results = pd.concat([free_living_results_matched_raw, combined]).reset_index(
    drop=True
)

algo_names = ws_level_results["algo_with_version"].unique()
cohort_names = ws_level_results["cohort"].unique()

ws_level_results["cohort"] = pd.Categorical(
    ws_level_results["cohort"], categories=cohort_names, ordered=True
)
ws_level_results["algo_with_version"] = pd.Categorical(
    ws_level_results["algo_with_version"], categories=algo_names, ordered=True
)

# Create the figure with subplots
fig = plt.figure(constrained_layout=True, figsize=(18, 3 * len(algo_names)))
subfigs = fig.subfigures(len(algo_names), 1, wspace=0.1, hspace=0.1)

# Define the min and max limits for x and y axes
min_max_x = calc_min_max_with_margin(ws_level_results["walking_speed_mps__reference"])
min_max_y = calc_min_max_with_margin(ws_level_results["walking_speed_mps__abs_rel_error"])

# Plotting each algorithm version
for subfig, (algo, data) in zip(
        subfigs, ws_level_results.groupby("algo_with_version", observed=True)
):
    subfig.suptitle(algo)
    subfig.supxlabel("Walking Speed (m/s)")
    subfig.supylabel("Absolute Relative Error (%)")

    # Create subplots for each cohort
    axs = subfig.subplots(1, len(cohort_names), sharex=True, sharey=True)

    for ax, (cohort, cohort_data) in zip(
            axs, data.groupby("cohort", observed=True)
    ):
        # Scatter plot for the cohort data
        sns.scatterplot(
            data=cohort_data,
            x="walking_speed_mps__reference",  # Reference walking speed
            y="walking_speed_mps__abs_rel_error",  # Absolute error
            ax=ax,
            alpha=0.3,
        )

        # Define bins for walking speed
        bins = np.arange(0, cohort_data["walking_speed_mps__reference"].max() + 0.05, 0.05)
        cohort_data["speed_bin"] = pd.cut(
            cohort_data["walking_speed_mps__reference"], bins=bins
        )

        # Calculate bin centers
        cohort_data["bin_center"] = cohort_data["speed_bin"].apply(
            lambda x: x.mid
        )

        # Calculate median error per bin and cohort
        binned_data = (
            cohort_data.groupby("bin_center", observed=True)["walking_speed_mps__abs_rel_error"]
            .median()
            .reset_index()
        )

        # Plot the median lines for each bin
        sns.scatterplot(
            data=binned_data,
            x="bin_center",
            y="walking_speed_mps__abs_rel_error",  # Median error
            ax=ax,
        )

        ax.set_title(cohort)
        ax.set_xlabel(None)
        ax.set_ylabel(None)

        # Set axis limits
        ax.set_xlim(*min_max_x)
        ax.set_ylim(*min_max_y)

fig.show()
