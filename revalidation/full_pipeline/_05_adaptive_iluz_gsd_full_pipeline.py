"""
.. _pipeline_val_adaptive_iluz_gsd:

Orientation-independent GSD options in the full pipeline
=======================================================

This analysis compares full-pipeline variants that use the same gait sequence detection algorithm for both the
healthy/mildly impaired and impaired sub-pipelines:

* :class:`~mobgap.gait_sequences.GsdIluz`
* :class:`~mobgap.gait_sequences.GsdIonescu`
* :class:`~mobgap.gait_sequences.GsdIluzAdaptiveGravity`

The main question is whether an orientation-independent GSD option would be feasible for the regular-walking cohorts.
Therefore, the analysis first shows all free-living cohorts and then focuses on ``HA``, ``COPD``, and ``CHF``, where
the default Mobilise-D pipeline uses ILUZ GSD.

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
    "Official_MobiliseD_Pipeline__adaptive_iluz_gs_all": (
        "Mobilise-D Pipeline",
        "GsdIluzAdaptiveGravity",
    ),
}

baseline_version = "GsdIluz"
candidate_versions = ["GsdIonescu", "GsdIluzAdaptiveGravity"]
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
_free_living_results_raw = {
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
    values=_free_living_results_raw,
    index_cols=free_living_index_cols,
    col_prefix_filter=None,
    convert_rel_error=True,
)

del _free_living_results, _free_living_results_raw

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


def mean_metric_table(
    data: pd.DataFrame,
    *,
    group_cols: list[str],
    analysis: str,
) -> pd.DataFrame:
    rows = []
    grouping = [*group_cols, "version"] if group_cols else ["version"]
    for group_key, group_df in data.groupby(grouping, sort=False):
        group_key = (group_key,) if isinstance(group_key, str) else group_key
        group_values = dict(zip(grouping, group_key))
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
    return pd.DataFrame(rows)


def paired_delta_table(
    data: pd.DataFrame,
    *,
    group_cols: list[str],
    analysis: str,
) -> pd.DataFrame:
    rows = []
    grouping = (
        data.groupby(group_cols, sort=False)
        if group_cols
        else [("All cohorts", data)]
    )
    for group_key, group_df in grouping:
        group_key = (group_key,) if isinstance(group_key, str) else group_key
        group_values = dict(zip(group_cols or ["cohort"], group_key))
        for dmo, dmo_label in dmos.items():
            for metric, metric_label in error_metrics.items():
                column = f"{dmo}__{metric}"
                pivot = group_df.pivot(
                    index=free_living_index_cols,
                    columns="version",
                    values=column,
                )
                for candidate in candidate_versions:
                    paired = pivot[[baseline_version, candidate]].dropna()
                    delta = paired[candidate] - paired[baseline_version]
                    rows.append(
                        {
                            **group_values,
                            "analysis": analysis,
                            "candidate": candidate,
                            "dmo": dmo_label,
                            "metric": metric_label,
                            "baseline_mean": paired[baseline_version].mean(),
                            "candidate_mean": paired[candidate].mean(),
                            "delta_mean": delta.mean(),
                            "delta_median": delta.median(),
                            "delta_std": delta.std(),
                            "n_recordings": len(paired),
                        }
                    )
    return pd.DataFrame(rows)


def paired_count_delta_table(
    data: pd.DataFrame,
    *,
    group_cols: list[str],
    analysis: str,
) -> pd.DataFrame:
    rows = []
    grouping = (
        data.groupby(group_cols, sort=False)
        if group_cols
        else [("All cohorts", data)]
    )
    for group_key, group_df in grouping:
        group_key = (group_key,) if isinstance(group_key, str) else group_key
        group_values = dict(zip(group_cols or ["cohort"], group_key))
        pivot = group_df.pivot(
            index=free_living_index_cols,
            columns="version",
            values="n_matched_wbs",
        )
        for candidate in candidate_versions:
            paired = pivot[[baseline_version, candidate]].dropna()
            delta = paired[candidate] - paired[baseline_version]
            rows.append(
                {
                    **group_values,
                    "analysis": analysis,
                    "candidate": candidate,
                    "baseline_mean": paired[baseline_version].mean(),
                    "candidate_mean": paired[candidate].mean(),
                    "delta_mean": delta.mean(),
                    "delta_median": delta.median(),
                    "delta_std": delta.std(),
                    "n_recordings": len(paired),
                }
            )
    return pd.DataFrame(rows)


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


# %%
# All cohorts
# -----------

combined_means_all = mean_metric_table(
    free_living_results_combined,
    group_cols=[],
    analysis="Combined",
)
matched_means_all = mean_metric_table(
    free_living_results_matched,
    group_cols=[],
    analysis="Matched",
)
combined_delta_all = paired_delta_table(
    free_living_results_combined,
    group_cols=[],
    analysis="Combined",
)
matched_delta_all = paired_delta_table(
    free_living_results_matched,
    group_cols=[],
    analysis="Matched",
)
matched_count_delta_all = paired_count_delta_table(
    free_living_results_matched,
    group_cols=[],
    analysis="Matched",
)

print("\nMean combined performance across all cohorts")
print(combined_means_all.round(4).to_string(index=False))
print("\nPaired combined deltas vs GsdIluz across all cohorts")
print(combined_delta_all.round(4).to_string(index=False))
print("\nMean matched performance across all cohorts")
print(matched_means_all.round(4).to_string(index=False))
print("\nPaired matched deltas vs GsdIluz across all cohorts")
print(matched_delta_all.round(4).to_string(index=False))
print("\nMatched WB count deltas vs GsdIluz across all cohorts")
print(matched_count_delta_all.round(4).to_string(index=False))

combined_means_all.round(4)

# %%
combined_delta_all.round(4)

# %%
matched_means_all.round(4)

# %%
matched_delta_all.round(4)

# %%
matched_count_delta_all.round(4)

# %%
# Per-cohort overview
# -------------------

combined_delta_cohort = paired_delta_table(
    free_living_results_combined,
    group_cols=["cohort"],
    analysis="Combined",
)
matched_delta_cohort = paired_delta_table(
    free_living_results_matched,
    group_cols=["cohort"],
    analysis="Matched",
)
matched_count_delta_cohort = paired_count_delta_table(
    free_living_results_matched,
    group_cols=["cohort"],
    analysis="Matched",
)

combined_delta_cohort = (
    combined_delta_cohort.set_index("cohort").loc[cohort_order].reset_index()
)
matched_delta_cohort = (
    matched_delta_cohort.set_index("cohort").loc[cohort_order].reset_index()
)
matched_count_delta_cohort = (
    matched_count_delta_cohort.set_index("cohort")
    .loc[cohort_order]
    .reset_index()
)

combined_delta_cohort.round(4)

# %%
matched_delta_cohort.round(4)

# %%
matched_count_delta_cohort.round(4)

# %%
# Absolute relative error deltas across all cohorts
# -------------------------------------------------

sns.set_context("talk")

abs_rel_delta = pd.concat(
    [
        paired_delta_long(
            free_living_results_combined,
            "abs_rel_error",
            analysis="Combined",
        ),
        paired_delta_long(
            free_living_results_matched,
            "abs_rel_error",
            analysis="Matched",
        ),
    ],
    ignore_index=True,
)

fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
for ax, (analysis_name, analysis_df) in zip(
    axes, abs_rel_delta.groupby("analysis", sort=False)
):
    ax.axhline(0, color="0.3", linewidth=1)
    sns.boxplot(
        data=analysis_df,
        x="cohort",
        y="delta",
        hue="candidate",
        order=cohort_order,
        hue_order=candidate_versions,
        showmeans=True,
        ax=ax,
    )
    ax.set_title(
        f"{analysis_name}: candidate - GsdIluz absolute relative error"
    )
    ax.set_ylabel("Delta abs. rel. error [%]")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title=None, loc="upper left")
axes[-1].set_xlabel("Cohort")
fig.tight_layout()
plt.show()

# %%
# Regular-walking cohorts
# -----------------------
# These are the cohorts where the default Mobilise-D pipeline uses ILUZ GSD. This is the key comparison for deciding
# whether an orientation-independent GSD can be used for the regular-walking pipeline.

combined_regular = free_living_results_combined[
    free_living_results_combined["cohort"].isin(regular_walking_cohorts)
].copy()
matched_regular = free_living_results_matched[
    free_living_results_matched["cohort"].isin(regular_walking_cohorts)
].copy()

combined_delta_regular_all = paired_delta_table(
    combined_regular,
    group_cols=[],
    analysis="Combined",
)
matched_delta_regular_all = paired_delta_table(
    matched_regular,
    group_cols=[],
    analysis="Matched",
)
matched_count_delta_regular_all = paired_count_delta_table(
    matched_regular,
    group_cols=[],
    analysis="Matched",
)

print("\nPaired combined deltas vs GsdIluz for HA/COPD/CHF")
print(combined_delta_regular_all.round(4).to_string(index=False))
print("\nPaired matched deltas vs GsdIluz for HA/COPD/CHF")
print(matched_delta_regular_all.round(4).to_string(index=False))
print("\nMatched WB count deltas vs GsdIluz for HA/COPD/CHF")
print(matched_count_delta_regular_all.round(4).to_string(index=False))

combined_delta_regular_all.round(4)

# %%
matched_delta_regular_all.round(4)

# %%
matched_count_delta_regular_all.round(4)

# %%
# Regular-walking cohorts by cohort
# ---------------------------------

combined_delta_regular_cohort = combined_delta_cohort[
    combined_delta_cohort["cohort"].isin(regular_walking_cohorts)
].copy()
matched_delta_regular_cohort = matched_delta_cohort[
    matched_delta_cohort["cohort"].isin(regular_walking_cohorts)
].copy()
matched_count_delta_regular_cohort = matched_count_delta_cohort[
    matched_count_delta_cohort["cohort"].isin(regular_walking_cohorts)
].copy()

combined_delta_regular_cohort.round(4)

# %%
matched_delta_regular_cohort.round(4)

# %%
matched_count_delta_regular_cohort.round(4)

# %%
# Regular-walking WB-level error distributions
# --------------------------------------------

regular_raw = free_living_results_matched_raw[
    free_living_results_matched_raw["cohort"].isin(regular_walking_cohorts)
].copy()

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
for ax, (dmo, dmo_label) in zip(axes, dmos.items()):
    sns.boxplot(
        data=regular_raw,
        x="cohort",
        y=f"{dmo}__abs_rel_error",
        hue="version",
        hue_order=version_order,
        order=regular_walking_cohorts,
        showmeans=True,
        ax=ax,
    )
    ax.set_title(dmo_label)
    ax.set_xlabel("Cohort")
    ax.set_ylabel("WB-level abs. rel. error [%]")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend().set_title(None)
fig.tight_layout()
plt.show()
