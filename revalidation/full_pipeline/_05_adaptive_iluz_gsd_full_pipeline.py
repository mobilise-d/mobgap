"""
.. _pipeline_val_adaptive_iluz_gsd:

Orientation-independent GSD options in the full pipeline
========================================================

This analysis compares full-pipeline variants that use the same gait sequence detection algorithm for both the
healthy/mildly impaired and impaired sub-pipelines:

* :class:`~mobgap.gait_sequences.GsdIluz`
* :class:`~mobgap.gait_sequences.GsdIonescu`
* :class:`~mobgap.gait_sequences.GsdIluzAdaptiveGravity`

``GsdIluz`` is currently the default algorithm for the regular-walking sub-pipeline, while ``GsdIonescu`` is the
default for the more impaired sub-pipeline.
While ``GsdIonescu`` is orientation-independent, ``GsdIluz`` is not, making it impossible to use the healthy
sub-pipeline without known fixed sensor orientation.
``GsdIluzAdaptiveGravity`` is a variant of ``GsdIluz`` that is specifically developed to be orientation-independent.

The goal of this analysis is to see, if it would be feasible to swap out ``GsdIluz`` for one of the
orientation-independent options in the regular-walking sub-pipeline.

The report starts with the regular-walking cohorts (HA, COPD, CHF), because these are the cohorts where replacing
``GsdIluz`` is most relevant. Results across all cohorts are shown afterwards for completeness.

.. note:: If you are interested in how these results are calculated, head over to the
    :ref:`processing page <pipeline_val_gen>`.

"""

from typing import Optional

# %%
# Compared pipelines
# ------------------

algorithms = {
    "Official_MobiliseD_Pipeline__gsd_iluz_all": (
        "Mobilise-D Pipeline",
        "GsdIluz",
    ),
    "Official_MobiliseD_Pipeline__gsd_ionescu_all": (
        "Mobilise-D Pipeline",
        "GsdIonescu",
    ),
    "Official_MobiliseD_Pipeline__gsd_iluz_adaptive_all": (
        "Mobilise-D Pipeline",
        "GsdIluzAdaptiveGravity",
    ),
}

baseline_version = "GsdIluz"
candidate_versions = ["GsdIluzAdaptiveGravity", "GsdIonescu"]
version_order = [baseline_version, *candidate_versions]
cohort_order = ["HA", "CHF", "COPD", "MS", "PD", "PFF"]
regular_walking_cohorts = ["HA", "COPD", "CHF"]

# %%
# Loading the free-living results
# -------------------------------
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mobgap.data.validation_results import ValidationResultLoader
from mobgap.plotting import (
    calc_min_max_with_margin,
    make_square,
    move_legend_outside,
)
from mobgap.utils.misc import get_env_var


def format_loaded_results(
    values: dict[tuple[str, str], pd.DataFrame],
    index_cols: list[str],
    col_prefix_filter: Optional[str],
    convert_rel_error: bool = False,
) -> pd.DataFrame:
    formatted = (
        pd.concat(values, names=["algo", "version", *index_cols])
        .pipe(
            lambda df: (
                df.filter(like=col_prefix_filter) if col_prefix_filter else df
            )
        )
        .reset_index()
        .assign(
            algo_with_version=lambda df: (
                df["algo"] + " (" + df["version"] + ")"
            ),
            _combined="combined",
        )
    )

    if col_prefix_filter:
        formatted.columns = formatted.columns.str.removeprefix(
            col_prefix_filter
        )

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
    "full_pipeline",
    result_path=local_data_path,
    version="main",
)

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

# %%
# Comparison helpers
# ------------------

dmos = {
    "walking_speed_mps": "Walking speed",
    "stride_length_m": "Stride length",
    "cadence_spm": "Cadence",
}
error_metrics = {
    "error": "Error",
    "abs_error": "Abs. error",
    "rel_error": "Rel. error [%]",
    "abs_rel_error": "Abs. rel. error [%]",
}
summary_statistics = ["mean", "median", "std", "n_recordings"]


def _as_tuple(value) -> tuple:
    return value if isinstance(value, tuple) else (value,)


def _sort_metric_rows(
    data: pd.DataFrame, index_cols: list[str]
) -> pd.DataFrame:
    data = data.copy()
    data["dmo"] = pd.Categorical(
        data["dmo"], categories=list(dmos.values()), ordered=True
    )
    data["metric"] = pd.Categorical(
        data["metric"], categories=list(error_metrics.values()), ordered=True
    )
    sort_cols = [*index_cols, "analysis", "dmo", "metric"]
    if "statistic" in data.columns:
        data["statistic"] = pd.Categorical(
            data["statistic"], categories=summary_statistics, ordered=True
        )
        sort_cols.append("statistic")
    data = data.sort_values(sort_cols).reset_index(drop=True)
    for col in ["dmo", "metric", "statistic"]:
        if col in data.columns:
            data[col] = data[col].astype(str)
    return data


def _with_index(data: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    data = data.set_index(index_cols)
    data.columns.name = None
    return data


def mean_metric_table(
    data: pd.DataFrame,
    *,
    group_cols: list[str],
    analysis: str,
) -> pd.DataFrame:
    rows = []
    grouping = [*group_cols, "version"] if group_cols else ["version"]
    for group_key, group_df in data.groupby(grouping, sort=False):
        group_values = dict(zip(grouping, _as_tuple(group_key)))
        for dmo, dmo_label in dmos.items():
            for metric, metric_label in error_metrics.items():
                column = f"{dmo}__{metric}"
                rows.append(
                    {
                        **group_values,
                        "analysis": analysis,
                        "dmo": dmo_label,
                        "metric": metric_label,
                        "mean": group_df[column].mean(),
                        "median": group_df[column].median(),
                        "std": group_df[column].std(),
                        "n_recordings": len(group_df[column].dropna()),
                    }
                )
    summary = pd.DataFrame(rows)
    index_cols = [*group_cols, "analysis", "dmo", "metric"]
    formatted = (
        summary.melt(
            id_vars=[*index_cols, "version"],
            value_vars=summary_statistics,
            var_name="statistic",
            value_name="value",
        )
        .pivot(
            index=[*index_cols, "statistic"],
            columns="version",
            values="value",
        )
        .reset_index()
    )
    formatted = formatted[[*index_cols, "statistic", *version_order]]
    return _with_index(
        _sort_metric_rows(formatted, group_cols),
        [*group_cols, "analysis", "dmo", "metric", "statistic"],
    )


def paired_metric_table(
    data: pd.DataFrame,
    *,
    group_cols: list[str],
    analysis: str,
    group_label: str = "All cohorts",
) -> pd.DataFrame:
    rows = []
    grouping = (
        data.groupby(group_cols, sort=False)
        if group_cols
        else [(group_label, data)]
    )
    for group_key, group_df in grouping:
        group_values = dict(zip(group_cols or ["cohort"], _as_tuple(group_key)))
        for dmo, dmo_label in dmos.items():
            for metric, metric_label in error_metrics.items():
                column = f"{dmo}__{metric}"
                pivot = group_df.pivot(
                    index=free_living_index_cols,
                    columns="version",
                    values=column,
                )
                paired = pivot[version_order].dropna()
                row = {
                    **group_values,
                    "analysis": analysis,
                    "dmo": dmo_label,
                    "metric": metric_label,
                    **{
                        version: paired[version].mean()
                        for version in version_order
                    },
                    "n_recordings": len(paired),
                }
                rows.append(row)
    index_cols = [*(group_cols or ["cohort"]), "analysis", "dmo", "metric"]
    result = pd.DataFrame(rows)[[*index_cols, *version_order, "n_recordings"]]
    return _with_index(
        _sort_metric_rows(result, group_cols or ["cohort"]),
        index_cols,
    )


def paired_delta_long(
    data: pd.DataFrame, metric: str, *, analysis: str
) -> pd.DataFrame:
    rows = []
    for dmo, dmo_label in dmos.items():
        column = f"{dmo}__{metric}"
        pivot = data.pivot(
            index=free_living_index_cols,
            columns="version",
            values=column,
        )
        for candidate in candidate_versions:
            paired = pivot[[baseline_version, candidate]].dropna()
            delta = paired[candidate] - paired[baseline_version]
            rows.append(
                delta.rename("delta")
                .reset_index()
                .assign(
                    candidate=candidate,
                    dmo=dmo_label,
                    metric=error_metrics[metric],
                    analysis=analysis,
                )
            )
    return pd.concat(rows, ignore_index=True)


sns.set_context("talk")


def plot_abs_rel_errors_by_cohort(
    data: pd.DataFrame,
    *,
    cohorts: list[str],
    ylabel: str,
    showfliers: bool = True,
) -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5), sharey=False, constrained_layout=True
    )
    for ax, (dmo, dmo_label) in zip(axes, dmos.items()):
        sns.boxplot(
            data=data,
            x="cohort",
            y=f"{dmo}__abs_rel_error",
            hue="version",
            hue_order=version_order,
            order=cohorts,
            showmeans=True,
            showfliers=showfliers,
            ax=ax,
        )
        ax.set_title(dmo_label)
        ax.set_xlabel("Cohort")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(title=None)
    move_legend_outside(fig, axes[-1])
    plt.show()


def matched_wb_count_table(
    data: pd.DataFrame, *, cohorts: list[str]
) -> pd.DataFrame:
    table = (
        data.groupby(["cohort", "version"], sort=False)["n_matched_wbs"]
        .sum()
        .unstack("version")
        .reindex(cohorts)
    )
    return table[version_order].astype(int).rename_axis(columns=None)


def plot_matched_wb_counts(data: pd.DataFrame, *, cohorts: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    sns.barplot(
        data=data.groupby(["version", "cohort"])["n_matched_wbs"]
        .sum()
        .reset_index(),
        hue="version",
        y="n_matched_wbs",
        x="cohort",
        order=cohorts,
        hue_order=version_order,
        ax=ax,
    )
    ax.set_ylabel("# Matched WBs")
    ax.set_xlabel("Cohort")
    ax.set_title("Matched WBs per cohort")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title=None)
    move_legend_outside(fig, ax)
    plt.show()


def plot_combined_walking_speed_error_correlation(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    column = "walking_speed_mps__error"
    pivoted_data = data.pivot(
        index=free_living_index_cols,
        columns="version",
        values=column,
    ).reset_index()[[baseline_version, *candidate_versions, "cohort"]]
    min_max = calc_min_max_with_margin(
        pivoted_data[baseline_version],
        *[pivoted_data[v] for v in candidate_versions],
    )

    for ax, candidate in zip(axes, candidate_versions):
        plot_data = pivoted_data[
            [baseline_version, candidate, "cohort"]
        ].dropna()
        sns.scatterplot(
            data=plot_data,
            x=baseline_version,
            y=candidate,
            hue="cohort",
            hue_order=regular_walking_cohorts,
            ax=ax,
        )
        make_square(ax, min_max, draw_diagonal=True)
        ax.axhline(0, color="0.4", linewidth=1, linestyle=":", zorder=-50)
        ax.axvline(0, color="0.4", linewidth=1, linestyle=":", zorder=-50)
        corr = plot_data[baseline_version].corr(plot_data[candidate])
        ax.set_title(f"{candidate} (r = {corr:.3f})")
        ax.set_xlabel(f"{baseline_version} error [m/s]")
        ax.set_ylabel(f"{candidate} error [m/s]")
        ax.grid(True, alpha=0.3)
        ax.legend(title=None)
    fig.suptitle(
        f"Combined walking-speed error agreement vs {baseline_version}"
    )
    move_legend_outside(fig, axes[-1])
    plt.show()


combined_regular = free_living_results_combined[
    free_living_results_combined["cohort"].isin(regular_walking_cohorts)
].copy()
matched_regular = free_living_results_matched[
    free_living_results_matched["cohort"].isin(regular_walking_cohorts)
].copy()
combined_comparison_all = paired_metric_table(
    free_living_results_combined,
    group_cols=[],
    analysis="Combined",
)
combined_comparison_cohort = (
    paired_metric_table(
        free_living_results_combined,
        group_cols=["cohort"],
        analysis="Combined",
    )
    .loc[cohort_order]
    .copy()
)

combined_comparison_regular_all = paired_metric_table(
    combined_regular,
    group_cols=[],
    analysis="Combined",
    group_label="HA/COPD/CHF",
)
combined_comparison_regular_cohort = (
    paired_metric_table(
        combined_regular,
        group_cols=["cohort"],
        analysis="Combined",
    )
    .loc[regular_walking_cohorts]
    .copy()
)

matched_comparison_regular_all = paired_metric_table(
    matched_regular,
    group_cols=[],
    analysis="Matched",
    group_label="HA/COPD/CHF",
)
matched_comparison_regular_cohort = (
    paired_metric_table(
        matched_regular,
        group_cols=["cohort"],
        analysis="Matched",
    )
    .loc[regular_walking_cohorts]
    .copy()
)

matched_wb_counts_regular = matched_wb_count_table(
    matched_regular, cohorts=regular_walking_cohorts
)

# %%
# Regular-walking cohorts: combined recording-level results
# ---------------------------------------------------------
# These are the cohorts where the default Mobilise-D pipeline uses ILUZ GSD. This is the key comparison for deciding
# whether an orientation-independent GSD can be used for the regular-walking pipeline.
#
# The combined analysis compares the median DMO values across each full recording between the wearable-derived
# pipeline output and the INDIP reference. This is the most relevant view for real-world performance, where downstream
# users typically consume recording-level aggregates rather than individual walking-bout estimates.

combined_comparison_regular_all.round(4)

# %%
combined_comparison_regular_cohort.round(4)

# %%
plot_abs_rel_errors_by_cohort(
    combined_regular,
    cohorts=regular_walking_cohorts,
    ylabel="Recording-level abs. rel. error [%]",
)

# %%
plot_abs_rel_errors_by_cohort(
    combined_regular,
    cohorts=regular_walking_cohorts,
    ylabel="Recording-level abs. rel. error [%]",
    showfliers=False,
)

# %%
# Regular-walking cohorts: direct walking-speed error comparison
# --------------------------------------------------------------
# These plots directly compare the combined walking-speed errors between ``GsdIluz`` and each orientation-independent
# candidate. The error is the detected walking speed minus the INDIP reference walking speed. Each point is one
# participant/recording, colored by cohort.

plot_combined_walking_speed_error_correlation(combined_regular)

# %%
# Regular-walking cohorts: matched recording-level results
# --------------------------------------------------------
# The matched analysis compares only walking bouts that overlap sufficiently between the wearable-derived output and
# the INDIP reference. This makes the DMO error comparison more like-for-like on the selected walking bouts, but it also
# depends on how many and which WBs each pipeline matched. It is therefore less representative for real-world
# performance, where recording-level aggregates over the entire recording are used.
#
# The first table and plot show the total number of matched WBs per cohort and pipeline. These counts should be used as
# context for interpreting the matched DMO error tables and plots below.

matched_wb_counts_regular

# %%
plot_matched_wb_counts(matched_regular, cohorts=regular_walking_cohorts)

# %%
matched_comparison_regular_all.round(4)

# %%
matched_comparison_regular_cohort.round(4)

# %%
plot_abs_rel_errors_by_cohort(
    matched_regular,
    cohorts=regular_walking_cohorts,
    ylabel="Recording-level matched-WB abs. rel. error [%]",
)

# %%
plot_abs_rel_errors_by_cohort(
    matched_regular,
    cohorts=regular_walking_cohorts,
    ylabel="Recording-level matched-WB abs. rel. error [%]",
    showfliers=False,
)

# %%
# All cohorts: combined recording-level results
# ---------------------------------------------
# For completeness, the combined analysis is also shown across all available cohorts. Matched results are intentionally
# omitted here, because the regular-walking cohorts above are the target use case for replacing ``GsdIluz``.

combined_comparison_all.round(4)

# %%
combined_comparison_cohort.round(4)

# %%
plot_abs_rel_errors_by_cohort(
    free_living_results_combined,
    cohorts=cohort_order,
    ylabel="Recording-level abs. rel. error [%]",
)
