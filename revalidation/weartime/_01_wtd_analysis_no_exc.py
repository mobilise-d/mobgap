"""
.. _wtd_val_results_no_exc:

Performance of the wear-time detection algorithms on the SUSTAIN dataset
========================================================================

This script analyses unpublished participant-LOSO results on SUSTAIN human
recordings and a fixed held-out half of simulated non-wear source recordings.
Each evaluated day contains at least eight hours of recorded data. The signal
detector has no learned parameters; LOSO does not undo historical tuning of
its fixed settings.

Classification summaries compare equal weights for days, participants and folds.
Within each participant, rates use either pooled labeled-sample confusion counts
or the mean of daily rates. Duration summaries give each non-missing daily error
equal weight. All means include symmetric 95% Student's t confidence intervals
using their averaging units. Intervals require at least two non-missing units
and are not clipped. The day-based intervals treat days as independent and do
not account for repeated days within a participant. Day counts and medians
remain point estimates.

.. note::
    See :ref:`wtd_val_gen_no_exc` for result generation. These results are not
    part of the published validation-result package. Only current result files
    containing both human and simulated non-wear populations are supported.

"""

# %%
# Compared algorithms
# -------------------
# At the moment, this local validation compares the signal-based wear-time
# detector implemented in MobGap. Additional
# algorithms can be added here once they are implemented as
# :class:`~mobgap.weartime.base.BaseWeartimeDetector`
# instances and evaluated with the wear-time result-generation script.
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from mobgap.pipeline.evaluation import ErrorTransformFuncs as E
from mobgap.utils.df_operations import CustomOperation, apply_transformations
from mobgap.utils.misc import get_env_var
from mobgap.weartime.evaluation import (
    calculate_wtd_classification_summary,
    calculate_wtd_simulated_non_wear_summary,
)
from scipy.stats import t

algorithms = {
    "WtdMegaritisSignal": ("WtdMegaritisSignal", "MobGap"),
}

results_base_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH"))
    / "results/weartime_loso_no_exc_min8h"
)
condition_name = "sustain_weartime"
index_cols = [
    "fold",
    "recording_type",
    "participant_id",
    "recording_id",
    "recording_day",
]


def load_single_results(result_name: str) -> pd.DataFrame:
    result_path = (
        results_base_path / condition_name / result_name / "single_results.csv"
    )
    if not result_path.exists():
        raise FileNotFoundError(
            f"Could not find generated wear-time results at {result_path}. "
            "Run revalidation/weartime/"
            "_02_wtd_result_generation_no_exc.py first."
        )

    results = pd.read_csv(result_path)
    available_index_cols = [
        column for column in index_cols if column in results.columns
    ]
    return results.set_index(available_index_cols)


results = pd.concat(
    {
        display_name: load_single_results(result_name)
        for result_name, display_name in algorithms.items()
    },
    names=["algo", "version"],
).reset_index()
human_movement_results = results[
    results["recording_type"] == "human_movement"
].copy()
simulated_non_wear_results = results[
    results["recording_type"] == "simulated_movements"
].copy()

# Use the same error functions as the cadence/stride-length evaluations.
# Relative errors are percentages here; zero reference duration makes them
# undefined.
for prefix in ("", "waking_"):
    reference_col = f"{prefix}reference_weartime_min"
    detected_col = f"{prefix}detected_weartime_min"
    transforms = [
        CustomOperation(
            identifier=None,
            function=partial(
                func,
                reference_col_name=reference_col,
                detected_col_name=detected_col,
                **kwargs,
            ),
            column_name=name,
        )
        for func, name, kwargs in (
            (E.error, f"{prefix}weartime_error_min", {}),
            (E.abs_error, f"abs_{prefix}weartime_error_min", {}),
            (
                E.rel_error,
                f"rel_{prefix}weartime_error_pct",
                {"zero_division_hint": np.nan},
            ),
            (
                E.abs_rel_error,
                f"abs_rel_{prefix}weartime_error_pct",
                {"zero_division_hint": np.nan},
            ),
        )
    ]
    errors = apply_transformations(human_movement_results, transforms)
    for column in errors.filter(like="_pct").columns:
        errors[column] *= 100
    human_movement_results[errors.columns] = errors

human_movement_results["algo_with_version"] = (
    human_movement_results["algo"]
    + " ("
    + human_movement_results["version"]
    + ")"
)

# Compare day, participant and fold weights using the daily confusion counts.
classification_overall = pd.DataFrame(
    {
        group: calculate_wtd_classification_summary(days)
        for group, days in human_movement_results.groupby(["algo", "version"])
    }
).T.rename_axis(index=["algo", "version"])
classification_by_participant = pd.DataFrame(
    {
        group: calculate_wtd_classification_summary(days)
        for group, days in human_movement_results.groupby(
            ["participant_id", "algo", "version"]
        )
    }
).T.rename_axis(index=["participant_id", "algo", "version"])

# %%
# Overview
# --------
# The core classification metrics are calculated from interval overlaps on the
# sample level. The duration metrics are
# reported in minutes. Waking-hours duration metrics use the same waking-hours
# configuration as the algorithm.
human_movement_summary_aggs = {
    "n_days": ("weartime_error_min", "size"),
    "reference_weartime_min_mean": ("reference_weartime_min", "mean"),
    "detected_weartime_min_mean": ("detected_weartime_min", "mean"),
    "weartime_error_min_mean": ("weartime_error_min", "mean"),
    "rel_weartime_error_pct_mean": ("rel_weartime_error_pct", "mean"),
    "abs_rel_weartime_error_pct_mean": ("abs_rel_weartime_error_pct", "mean"),
    "rel_waking_weartime_error_pct_mean": (
        "rel_waking_weartime_error_pct",
        "mean",
    ),
    "abs_rel_waking_weartime_error_pct_mean": (
        "abs_rel_waking_weartime_error_pct",
        "mean",
    ),
    "weartime_error_min_median": ("weartime_error_min", "median"),
    "abs_weartime_error_min_mean": ("abs_weartime_error_min", "mean"),
    "waking_reference_weartime_min_mean": (
        "waking_reference_weartime_min",
        "mean",
    ),
    "waking_detected_weartime_min_mean": (
        "waking_detected_weartime_min",
        "mean",
    ),
    "waking_weartime_error_min_mean": ("waking_weartime_error_min", "mean"),
    "abs_waking_weartime_error_min_mean": (
        "abs_waking_weartime_error_min",
        "mean",
    ),
}

human_movement_summary_overall = (
    human_movement_results.groupby(["algo", "version"])
    .agg(**human_movement_summary_aggs)
    .join(classification_overall)
)

# %%
# Human movement: per participant
# -------------------------------
# For the human-movement recordings we inspect the full set of overlap and
# duration metrics. The dataset is evaluated
# per day. Duration summaries average days; classification summaries compare
# pooled participant counts with average daily rates.
human_movement_summary_by_participant = (
    human_movement_results.groupby(["participant_id", "algo", "version"])
    .agg(**human_movement_summary_aggs)
    .join(classification_by_participant)
)

# Keep displayed fold summaries restricted to human recordings as well.
classification_by_fold = pd.DataFrame(
    {
        group: calculate_wtd_classification_summary(days)
        for group, days in human_movement_results.groupby(
            ["fold", "algo", "version"]
        )
    }
).T.rename_axis(index=["fold", "algo", "version"])
human_movement_summary_by_fold = (
    human_movement_results.groupby(["fold", "algo", "version"])
    .agg(**human_movement_summary_aggs)
    .join(classification_by_fold)
)

# Daily t intervals apply to duration means; counts and medians stay unchanged.
for group_columns, summary in (
    (["algo", "version"], human_movement_summary_overall),
    (["fold", "algo", "version"], human_movement_summary_by_fold),
    (
        ["participant_id", "algo", "version"],
        human_movement_summary_by_participant,
    ),
):
    grouped_days = human_movement_results.groupby(group_columns)
    for name, (column, aggregation) in human_movement_summary_aggs.items():
        if aggregation != "mean":
            continue
        units = grouped_days[column]
        half_width = units.sem() * t.ppf(0.975, units.count() - 1)
        summary[f"{name}__ci95_lower"] = summary[name] - half_width
        summary[f"{name}__ci95_upper"] = summary[name] + half_width

# %%
# Overall summary with confidence intervals
# -----------------------------------------
human_movement_summary_overall

# %%
# Participant summaries with confidence intervals
# ------------------------------------------------
human_movement_summary_by_participant

# %%
# Human movement: per fold
# -------------------------
human_movement_summary_by_fold

# %%
# Human movement plots
# --------------------
import matplotlib.pyplot as plt
import seaborn as sns

fig_human, axes = plt.subplots(2, 2, figsize=(13, 9))

sns.boxplot(
    data=human_movement_results,
    x="algo_with_version",
    y="abs_weartime_error_min",
    ax=axes[0, 0],
    showmeans=True,
)
axes[0, 0].set_title("Absolute overall wear-time error")
axes[0, 0].set_xlabel("")

sns.boxplot(
    data=human_movement_results,
    x="algo_with_version",
    y="abs_waking_weartime_error_min",
    ax=axes[0, 1],
    showmeans=True,
)
axes[0, 1].set_title("Absolute waking-hours wear-time error")
axes[0, 1].set_xlabel("")

sns.boxplot(
    data=human_movement_results,
    x="algo_with_version",
    y="f1_score",
    ax=axes[1, 0],
    showmeans=True,
)
axes[1, 0].set_title("F1 score")
axes[1, 0].set_xlabel("")

sns.scatterplot(
    data=human_movement_results,
    x="reference_weartime_min",
    y="detected_weartime_min",
    hue="algo_with_version",
    ax=axes[1, 1],
)
axes[1, 1].set_title("Detected vs reference wear-time")
axes[1, 1].set_xlabel("Reference wear-time [min]")
axes[1, 1].set_ylabel("Detected wear-time [min]")

for ax in axes.flatten():
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("")

plt.tight_layout()
fig_human.show()


# %%
# Simulated non-wear: independent source recordings
# -------------------------------------------------
# Pool sample counts within each recording/fold; false wear is the mean minutes
# per evaluated day, without normalization to 24 hours. Average fold models for
# each recording before averaging recordings and calculating the 95% t interval.
# Repeated predictions across folds do not add independent CI observations.
simulated_non_wear_summary_overall = pd.DataFrame(
    {
        group: calculate_wtd_simulated_non_wear_summary(days)
        for group, days in simulated_non_wear_results.groupby(
            ["algo", "version"]
        )
    }
).T.rename_axis(index=["algo", "version"])

recording_fold_groups = simulated_non_wear_results.groupby(
    ["algo", "version", "fold", "recording_id"]
)
recording_fold_counts = recording_fold_groups[
    ["tp_samples", "fp_samples", "fn_samples", "tn_samples"]
].sum()
simulated_non_wear_recording_fold_results = pd.DataFrame(
    {
        "combined__specificity": recording_fold_counts["tn_samples"]
        / (
            recording_fold_counts["tn_samples"]
            + recording_fold_counts["fp_samples"]
        ),
        "combined__accuracy": (
            recording_fold_counts["tp_samples"]
            + recording_fold_counts["tn_samples"]
        )
        / recording_fold_counts.sum(axis=1),
        "day_mean__false_wear_min": recording_fold_groups[
            "detected_weartime_min"
        ].mean(),
    }
)
simulated_non_wear_recording_results = (
    simulated_non_wear_recording_fold_results.groupby(
        level=["algo", "version", "recording_id"]
    ).mean()
)
simulated_non_wear_summary_by_fold = (
    simulated_non_wear_recording_fold_results.groupby(
        level=["algo", "version", "fold"]
    ).mean()
)
# This descriptive SD measures model variation, not independent recording error.
# It is zero for the fixed signal detector's repeated predictions.
simulated_non_wear_model_sd = simulated_non_wear_recording_fold_results.groupby(
    level=["algo", "version", "recording_id"]
).std()

# %%
# Simulated non-wear summary with recording-level confidence intervals
# --------------------------------------------------------------------
simulated_non_wear_summary_overall

# %%
# Simulated non-wear: per recording, averaged across fold models
# -------------------------------------------------------------
simulated_non_wear_recording_results

# %%
# Simulated non-wear: per fold
# ----------------------------
simulated_non_wear_summary_by_fold

# %%
# Simulated non-wear: variation across fold models
# ------------------------------------------------
simulated_non_wear_model_sd

# The raw fold CSV retains the scorer's mixed-population aggregates for its
# existing low-level contract. Use the separate population tables above for
# performance comparisons; there is no combined human/non-wear primary score.
