"""Class to validate gait sequence detection results."""

import warnings
from typing import Any, Literal, Union, Unpack

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mobgap.gait_sequences._evaluation_scorer import (
    gsd_final_agg,
    gsd_per_datapoint_score,
    gsd_score,
)
from mobgap.utils.evaluation import (
    accuracy_score,
    npv_score,
    precision_recall_f1_score,
    specificity_score,
)
from mobgap.utils.intervals import (
    _check_gs_level_matches_sanity,  # noqa: F401 - compatibility re-export
    _check_input_sanity,  # noqa: F401 - compatibility re-export
    _get_false_matches_from_overlap_data,  # noqa: F401 - compatibility re-export
    _get_tn_intervals,  # noqa: F401 - compatibility re-export
    _sanitize_index,  # noqa: F401 - compatibility re-export
    categorize_intervals,
    categorize_intervals_per_sample,
    count_samples_in_intervals,
    count_samples_in_match_intervals,
    get_matching_intervals,
)


def calculate_matched_gsd_performance_metrics(
    matches: pd.DataFrame, *, zero_division: Literal["warn", 0, 1] = "warn"
) -> dict[str, Union[float, int]]:
    """
    Calculate commonly known performance metrics for based on the matched overlap with the reference.

    This method assumes that you already calculated the overlapping regions between the ground truth and the detected
    gait sequences using the :func:`~mobgap.utils.intervals.categorize_intervals_per_sample` method.
    This function then calculates the performance metrics based on the number of true positive, false positive,
    false negative, and true negative matches between detected and reference.
    All calculations are performed on the sample level.
    This means the intervals provided by the input dataframe are used to calculate the number of samples in each
    category.

    The following metrics are always returned:

    - `tp_samples`: Number of samples that are correctly detected as gait sequences.
    - `fp_samples`: Number of samples that are falsely detected as gait sequences.
    - `fn_samples`: Number of samples that are not detected as gait sequences.
    - `precision`: Precision of the detected gait sequences.
    - `recall`: Recall of the detected gait sequences.
    - `f1_score`: F1 score of the detected gait sequences.
    - `tn_samples`: Number of samples that are correctly not detected as gait sequences.
    - `specificity`: Specificity of the detected gait sequences.
    - `accuracy`: Accuracy of the detected gait sequences.
    - `npv`: Negative predictive value of the detected gait sequences.

    See the documentation of :func:`~mobgap.utils.evaluation.precision_recall_f1_score` for more details.

    Note, that some of the metrics might run into zero division errors depending on the input data.
    The resulting value can be controlled by the `zero_division` parameter.
    See the documentation of :func:`~mobgap.utils.evaluation.specificity_score`,
    :func:`~mobgap.utils.evaluation.accuracy_score`, and :func:`~mobgap.utils.evaluation.npv_score` for more
    details.
    Note, that you can only control the behavior of the zero division for all metrics at once.
    If you need individual control, you need to calculate the metrics separately.

    Parameters
    ----------
    matches: pd.DataFrame
        A DataFrame as returned by :func:`~mobgap.utils.intervals.categorize_intervals_per_sample`.
        It contains the matched intervals between algorithm output and reference with their `start` and `end` index
        and the respective `match_type`.
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    gsd_metrics: dict

    See Also
    --------
    calculate_unmatched_gsd_performance_metrics
        For calculating performance metrics without matching the detected and reference gait sequences.
    categorize_intervals_per_sample
        For categorizing the detected and reference gait sequences on a sample-wise level.
        This is required to calculate the ``matches`` parameter for this function.

    """
    matches = _check_sample_level_matches_sanity(matches)

    tp_samples = count_samples_in_match_intervals(matches, match_type="tp")
    fp_samples = count_samples_in_match_intervals(matches, match_type="fp")
    fn_samples = count_samples_in_match_intervals(matches, match_type="fn")
    tn_samples = count_samples_in_match_intervals(matches, match_type="tn")

    # estimate performance metrics
    precision_recall_f1 = precision_recall_f1_score(matches, zero_division=zero_division)

    gsd_metrics = {
        "tp_samples": tp_samples,
        "fp_samples": fp_samples,
        "fn_samples": fn_samples,
        **precision_recall_f1,
        "tn_samples": tn_samples,
        "specificity": specificity_score(matches, zero_division=zero_division),
        "accuracy": accuracy_score(matches, zero_division=zero_division),
        "npv": npv_score(matches, zero_division=zero_division),
    }
    return gsd_metrics


def calculate_unmatched_gsd_performance_metrics(
    *,
    gsd_list_detected: pd.DataFrame,
    gsd_list_reference: pd.DataFrame,
    sampling_rate_hz: float,
    zero_division_hint: Union[Literal["warn", "raise"], float] = "warn",
) -> dict[str, Union[float, int]]:
    """
    Calculate general performance metrics that don't rely on matching the detected and reference gait sequences.

    Metrics calculated by this function are just based on the overall amount of gait detected.
    The following metrics are calculated:

    - `reference_gs_duration_s`: Total duration of the reference gait sequences in seconds.
    - `detected_gs_duration_s`: Total duration of the detected gait sequences in seconds.
    - `gs_duration_error_s`: Difference between the detected and reference gait sequence duration in seconds.
    - `gs_relative_duration_error`: gs_duration_error_s divided by reference_gs_duration_s.
    - `gs_absolute_duration_error_s`: Absolute value of gs_duration_error_s.
    - `gs_absolute_relative_duration_error`: gs_absolute_duration_error_s divided by reference_gs_duration_s.
    - `gs_absolute_relative_duration_error_log`: np.log(1 + gs_duration_error_abs_rel)
    - `detected_num_gs`: Total number of detected gait sequences.
    - `reference_num_gs`: Total number of reference gait sequences.
    - `num_gs_error`: Difference between the detected and reference number of gait sequences.
    - `num_gs_relative_error`: num_gs_error divided by reference_num_gs.
    - `num_gs_absolute_error`: Absolute value of num_gs_error.
    - `num_gs_absolute_relative_error`: num_gs_absolute_error divided by reference_num_gs.
    - `num_gs_absolute_relative_error_log`: np.log(1 + num_gs_absolute_relative_error)

    Parameters
    ----------
    gsd_list_detected
       Each row contains a detected gait sequence interval as output from the GSD algorithms.
       The respective start index is stored in a column named `start` and the stop index in a column named `end`.
    gsd_list_reference
       Gold standard to validate the detected gait sequences against.
       Should have the same format as `gsd_list_detected`.
    sampling_rate_hz
        Sampling frequency of the recording in Hz.
    zero_division_hint : "warn", "raise" or np.nan, default="warn"
        Controls the behavior when there is a zero division. If set to "warn",
        affected metrics are set to NaN and a warning is raised.
        If set to "raise", a ZeroDivisionError is raised.
        If set to `np.nan`, the warning is suppressed and the affected metrics are set to NaN.
        Zero division can occur if there are no gait sequences in the reference data, i.e., reference_gs_duration_s = 0

    Returns
    -------
    gsd_metrics: dict

    See Also
    --------
    calculate_matched_gsd_performance_metrics
        For calculating performance metrics based on the matched overlap with the reference.
    categorize_intervals_per_sample
        For categorizing the detected and reference gait sequences on a sample-wise level.

    """
    if sampling_rate_hz <= 0:
        raise ValueError("The sampling rate must be larger than 0.")

    # estimate basic duration metrics
    reference_gs_duration_s = count_samples_in_intervals(gsd_list_reference) / sampling_rate_hz
    detected_gs_duration_s = count_samples_in_intervals(gsd_list_detected) / sampling_rate_hz
    gs_duration_error_s = detected_gs_duration_s - reference_gs_duration_s
    gs_absolute_duration_error_s = abs(gs_duration_error_s)

    # estimate basic gs count metrics
    detected_num_gs = len(gsd_list_detected)
    reference_num_gs = len(gsd_list_reference)
    num_gs_error = detected_num_gs - reference_num_gs
    num_gs_absolute_error = abs(num_gs_error)

    # check if reference gs are present to prevent zero division
    if reference_gs_duration_s == 0:
        if zero_division_hint not in ["warn", "raise", np.nan]:
            raise ValueError('"zero_division" must be set to "warn", "raise" or `np.nan`!')
        if zero_division_hint == "raise":
            raise ZeroDivisionError(
                "Zero division occurred because no gait sequences were detected in the reference data."
            )
        if zero_division_hint == "warn":
            warnings.warn(
                "Zero division occurred because no gait sequences were detected in the reference data. "
                "Affected metrics are set to NaN.",
                UserWarning,
                stacklevel=2,
            )
        gs_relative_duration_error = np.nan
        gs_absolute_relative_duration_error = np.nan
        num_gs_relative_error = np.nan
        num_gs_absolute_relative_error = np.nan
    # no zero division, calculate relative metrics
    else:
        gs_relative_duration_error = np.array(gs_duration_error_s) / reference_gs_duration_s
        gs_absolute_relative_duration_error = np.array(gs_absolute_duration_error_s) / reference_gs_duration_s
        num_gs_relative_error = num_gs_error / np.array(reference_num_gs)
        num_gs_absolute_relative_error = num_gs_absolute_error / np.array(reference_num_gs)

    # logarithmic relative metrics
    gs_absolute_relative_duration_error_log = np.log(1 + gs_absolute_relative_duration_error)
    num_gs_absolute_relative_error_log = np.log(1 + num_gs_absolute_relative_error)

    gsd_metrics = {
        "reference_gs_duration_s": reference_gs_duration_s,
        "detected_gs_duration_s": detected_gs_duration_s,
        "gs_duration_error_s": gs_duration_error_s,
        "gs_relative_duration_error": gs_relative_duration_error,
        "gs_absolute_duration_error_s": gs_absolute_duration_error_s,
        "gs_absolute_relative_duration_error": gs_absolute_relative_duration_error,
        "gs_absolute_relative_duration_error_log": gs_absolute_relative_duration_error_log,
        "detected_num_gs": detected_num_gs,
        "reference_num_gs": reference_num_gs,
        "num_gs_error": num_gs_error,
        "num_gs_relative_error": num_gs_relative_error,
        "num_gs_absolute_error": num_gs_absolute_error,
        "num_gs_absolute_relative_error": num_gs_absolute_relative_error,
        "num_gs_absolute_relative_error_log": num_gs_absolute_relative_error_log,
    }

    # Convert all np.floats to floats
    gsd_metrics = {k: float(v) if isinstance(v, np.float64) else v for k, v in gsd_metrics.items()}

    return gsd_metrics


def _check_sample_level_matches_sanity(matches: pd.DataFrame) -> pd.DataFrame:
    # check if input is a dataframe
    if not isinstance(matches, pd.DataFrame):
        raise TypeError("`matches` must be of type `pandas.DataFrame`.")
    # check for correct columns
    try:
        matches = matches[["start", "end", "match_type"]]
    except KeyError as e:
        raise ValueError("`matches` must have columns named `start`, `end`, and `match_type`.") from e
    # check if `match_type` column contains only valid values
    if not matches["match_type"].isin(["tp", "fp", "fn", "tn"]).all():
        raise ValueError("`match_type` must contain only the values 'tp', 'fp', 'fn', and 'tn'.")
    return matches


def plot_categorized_intervals(
    gsd_list_detected: pd.DataFrame, gsd_list_reference: pd.DataFrame, categorized_intervals: pd.DataFrame
) -> Figure:
    """Plot the categorized intervals together with the detected and reference intervals."""
    fig, ax = plt.subplots(figsize=(10, 3))
    _plot_intervals_from_df(gsd_list_reference, 3, ax, color="orange")
    _plot_intervals_from_df(gsd_list_detected, 2, ax, color="blue")
    _plot_intervals_from_df(categorized_intervals.query("match_type == 'tp'"), 1, ax, color="green", label="TP")
    _plot_intervals_from_df(categorized_intervals.query("match_type == 'fp'"), 1, ax, color="red", label="FP")
    _plot_intervals_from_df(categorized_intervals.query("match_type == 'fn'"), 1, ax, color="purple", label="FN")
    plt.yticks([1, 2, 3], ["Categorized", "Detected", "Reference"])
    plt.ylim(0, 4)
    plt.xlabel("Index")
    leg = plt.legend(loc="upper right", bbox_to_anchor=(1, 1.2), ncol=3, frameon=False)
    for handle in leg.legend_handles:
        handle.set_linewidth(10)
    plt.tight_layout()
    return fig


def _plot_intervals_from_df(df: pd.DataFrame, y: int, ax: Axes, **kwargs: Unpack[dict[str, Any]]) -> None:
    label_set = False
    for _, row in df.iterrows():
        label = kwargs.pop("label", None)
        if label and not label_set:
            ax.hlines(y, row["start"], row["end"], lw=20, label=label, **kwargs)
            label_set = True
        else:
            ax.hlines(y, row["start"], row["end"], lw=20, **kwargs)


__all__ = [
    "calculate_matched_gsd_performance_metrics",
    "calculate_unmatched_gsd_performance_metrics",
    "categorize_intervals",
    "categorize_intervals_per_sample",
    "get_matching_intervals",
    "gsd_final_agg",
    "gsd_per_datapoint_score",
    "gsd_score",
    "plot_categorized_intervals",
]
