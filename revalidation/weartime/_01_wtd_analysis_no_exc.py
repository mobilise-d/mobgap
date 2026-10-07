"""
.. _wtd_val_results_no_exc:

Performance of the wear-time detection algorithms on the SUSTAIN dataset
========================================================================

This script analyses unpublished participant-LOSO results on SUSTAIN human
recordings. Each held-out day contains at least eight hours of recorded data.
Part B is not evaluated. The signal detector has no learned parameters; LOSO
does not undo historical tuning of its fixed settings.

Classification metrics pool labeled-sample confusion counts across all held-out
participants. Duration summaries give each non-missing daily error equal weight,
rather than giving each participant fold equal weight.

.. note::
    See :ref:`wtd_val_gen_no_exc` for result generation. These results are not
    part of the published validation-result package.

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
from mobgap.gait_sequences.evaluation import (
    calculate_matched_gsd_performance_metrics,
)
from mobgap.pipeline.evaluation import ErrorTransformFuncs as E
from mobgap.utils.df_operations import CustomOperation, apply_transformations
from mobgap.utils.misc import get_env_var

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
    errors = apply_transformations(results, transforms)
    for column in errors.filter(like="_pct").columns:
        errors[column] *= 100
    results[errors.columns] = errors

results["algo_with_version"] = results["algo"] + " (" + results["version"] + ")"
human_movement_results = results[results["recording_type"] == "human_movement"]

# Pool raw held-out sample matches, not per-fold classification rates.
held_out_matches = pd.concat(
    {
        display_name: pd.read_csv(
            results_base_path / condition_name / result_name / "raw_matches.csv"
        )
        for result_name, display_name in algorithms.items()
    },
    names=["algo", "version"],
).reset_index()
pooled_classification_overall = pd.DataFrame(
    {
        group: calculate_matched_gsd_performance_metrics(
            matches.assign(end=matches["end"] - 1)
        )
        for group, matches in held_out_matches.groupby(["algo", "version"])
    }
).T.rename_axis(index=["algo", "version"])
pooled_classification_by_participant = pd.DataFrame(
    {
        group: calculate_matched_gsd_performance_metrics(
            matches.assign(end=matches["end"] - 1)
        )
        for group, matches in held_out_matches.groupby(
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
    .join(pooled_classification_overall)
)
human_movement_summary_overall

# %%
# Human movement: per participant
# -------------------------------
# For the human-movement recordings we inspect the full set of overlap and
# duration metrics. The dataset is evaluated
# per day; this aggregation keeps the participant identity, then averages across
# the selected participant's days.
human_movement_summary_by_participant = (
    human_movement_results.groupby(["participant_id", "algo", "version"])
    .agg(**human_movement_summary_aggs)
    .join(pooled_classification_by_participant)
)
human_movement_summary_by_participant

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
