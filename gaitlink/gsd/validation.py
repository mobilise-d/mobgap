"""Class to validate gait sequence detection results."""
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from intervaltree import IntervalTree
from intervaltree.interval import Interval
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing_extensions import Unpack

__all__ = [
    "categorize_intervals",
    "find_matches_with_min_overlap",
    "calculate_gsd_performance_metrics",
    "plot_categorized_intervals",
]

from gaitlink.utils.evaluation import (
    accuracy_score,
    count_samples_in_intervals,
    count_samples_in_match_intervals,
    npv_score,
    precision_recall_f1_score,
    specificity_score,
)


def calculate_gsd_performance_metrics(
    gsd_list_detected: pd.DataFrame,
    gsd_list_reference: pd.DataFrame,
    sampling_rate_hz: float,
    n_overall_samples: Union[int, None] = None,
) -> dict[str, float]:
    """
    Calculate performance metrics for gait sequence detection results.

    The detected and reference dataframes are expected to have columns named "start" and "end" containing the
    start and end indices of the respective gait sequences.

    The following metrics are calculated:

    - General metrics:
        - `tp_samples`: Number of samples that are correctly detected as gait sequences.
        - `fp_samples`: Number of samples that are falsely detected as gait sequences.
        - `fn_samples`: Number of samples that are not detected as gait sequences.
        - `precision`: Precision of the detected gait sequences.
        - `recall`: Recall of the detected gait sequences.
        - `f1_score`: F1 score of the detected gait sequences.
        See the documentation of :func:`~gaitlink.utils.evaluation.precision_recall_f1_score` for more details.

    - Mobilised-specific metrics:
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

    Further metrics are calculated if `n_samples` is provided and thus the number of true negatives can be estimated:

    - `tn_samples`: Number of samples that are correctly not detected as gait sequences.
    - `specificity`: Specificity of the detected gait sequences.
    - `accuracy`: Accuracy of the detected gait sequences.
    - `npv`: Negative predictive value of the detected gait sequences.
    See the documentation of :func:`~gaitlink.utils.evaluation.specificity_score`,
    :func:`~gaitlink.utils.evaluation.accuracy_score`,
    and :func:`~gaitlink.utils.evaluation.npv_score` for more details.


    Parameters
    ----------
    gsd_list_detected: pd.DataFrame
       Each row contains a detected gait sequence interval as output from the GSD algorithms.
       The respective start index is stored in a column named `start` and the stop index in a column named `end`.
    gsd_list_reference: pd.DataFrame
       Gold standard to validate the detected gait sequences against.
       Should have the same format as `gsd_list_detected`.
    sampling_rate_hz: float
        Sampling frequency of the recording in Hz.
    n_overall_samples: int, optional
        Number of samples in the analyzed recording.

    Returns
    -------
    gsd_metrics: dict
    """
    categorized_intervals = categorize_intervals(gsd_list_detected, gsd_list_reference)

    tn_intervals = _get_tn_intervals(categorized_intervals, n_samples=n_overall_samples)
    categorized_intervals = pd.concat([categorized_intervals, tn_intervals], ignore_index=True)
    categorized_intervals = categorized_intervals.sort_values(by=["start", "end"], ignore_index=True)

    if n_overall_samples and n_overall_samples < max(gsd_list_reference["end"].max(), gsd_list_detected["end"].max()):
        raise ValueError(
            "The provided `n_samples` parameter is implausible. The number of samples must be larger than the highest "
            "index in the detected and reference gait sequences."
        )

    # estimate tp, fp, fn
    tp_samples = count_samples_in_match_intervals(categorized_intervals, match_type="tp")
    fp_samples = count_samples_in_match_intervals(categorized_intervals, match_type="fp")
    fn_samples = count_samples_in_match_intervals(categorized_intervals, match_type="fn")

    # estimate performance metrics
    precision_recall_f1 = precision_recall_f1_score(categorized_intervals)

    # estimate duration metrics
    reference_gs_duration_s = count_samples_in_intervals(gsd_list_reference) / sampling_rate_hz
    detected_gs_duration_s = count_samples_in_intervals(gsd_list_detected) / sampling_rate_hz
    gs_duration_error_s = detected_gs_duration_s - reference_gs_duration_s
    gs_relative_duration_error = gs_duration_error_s / reference_gs_duration_s
    gs_absolute_duration_error_s = abs(gs_duration_error_s)  # TODO is it necessary to report absolute values here?
    gs_absolute_relative_duration_error = gs_absolute_duration_error_s / reference_gs_duration_s
    gs_absolute_relative_duration_error_log = 1 + np.log(
        1 + gs_absolute_relative_duration_error
    )  # TODO should this be included? Is the 1 + ... a typo?

    # estimate gs count metrics
    detected_num_gs = len(gsd_list_detected)
    reference_num_gs = len(gsd_list_reference)
    num_gs_error = detected_num_gs - reference_num_gs
    num_gs_relative_error = num_gs_error / reference_num_gs
    num_gs_absolute_error = abs(num_gs_error)  # TODO is it necessary to report absolute values here?
    num_gs_absolute_relative_error = num_gs_absolute_error / reference_num_gs
    num_gs_absolute_relative_error_log = np.log(1 + num_gs_absolute_relative_error)  # TODO should this be included?

    result = {
        "tp_samples": tp_samples,
        "fp_samples": fp_samples,
        "fn_samples": fn_samples,
        **precision_recall_f1,
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

    # tn-dependent metrics
    if n_overall_samples is not None:
        result["tn_samples"] = n_overall_samples
        result["specificity"] = specificity_score(categorized_intervals)
        result["accuracy"] = accuracy_score(categorized_intervals)
        result["npv"] = npv_score(categorized_intervals)

    return result


def categorize_intervals(gsd_list_detected: pd.DataFrame, gsd_list_reference: pd.DataFrame) -> pd.DataFrame:
    """
    Validate detected gait sequence intervals against a reference on a sample-wise level.

    The detected and reference dataframes are expected to have columns namend "start" and "end" containing the
    start and end indices of the respective gait sequences.
    Each sample from the detected interval list is categorized as true positive (tp),
    false positive (fp) or false negative (fn).
    The results are concatenated into intervals of tp, fp, and fn matches and returned as a DataFrame.

    Parameters
    ----------
    gsd_list_detected: pd.DataFrame
       Each row contains a detected gait sequence interval as output from the GSD algorithms.
       The respective start index is stored in a column named `start` and the stop index in a column named `end`.
    gsd_list_reference: pd.DataFrame
       Gold standard to validate the detected gait sequences against.
       Should have the same format as `gsd_list_detected`.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the categorized intervals with their `start` and `end` index
        and the respective `match_type`. Keep in mind that the intervals are not identical to the intervals
        in `gsd_list_detected`, but are rather split into subsequences according to their match type with the reference.

    Examples
    --------
    >>> from gaitlink.gsd.validation import categorize_intervals
    >>> detected = pd.DataFrame([[0, 10], [20, 30]], columns=["start", "end"])
    >>> reference = pd.DataFrame([[0, 10], [15, 25]], columns=["start", "end"])
    >>> result = categorize_intervals(detected, reference)
    >>> result.tp_intervals
           start  end match_type
    0      0   10         tp
    1     15   20         fn
    2     20   25         tp
    3     25   30         fp
    """
    detected, reference = _check_input_sanity(gsd_list_detected, gsd_list_reference)

    # Create Interval Trees
    reference_tree = IntervalTree.from_tuples(reference.to_numpy())
    detected_tree = IntervalTree.from_tuples(detected.to_numpy())

    # Prepare DataFrames for TP, FP, FN
    tp_intervals = []
    fp_intervals = []
    fn_intervals = []

    # Calculate TP and FP
    for interval in detected_tree:
        overlaps = sorted(reference_tree.overlap(interval.begin, interval.end))
        if overlaps:
            fp_matches = _get_false_matches_from_overlap_data(overlaps, interval)
            fp_intervals.extend(fp_matches)

            for overlap in overlaps:
                start = max(interval.begin, overlap.begin)
                end = min(interval.end, overlap.end)
                tp_intervals.append([start, end])

        else:
            fp_intervals.append([interval.begin, interval.end])

    # Calculate FN
    for interval in reference_tree:
        overlaps = sorted(detected_tree.overlap(interval.begin, interval.end))
        if not overlaps:
            fn_intervals.append([interval.begin, interval.end])
        else:
            fn_matches = _get_false_matches_from_overlap_data(overlaps, interval)
            fn_intervals.extend(fn_matches)

    # convert results to pandas DataFrame and add a match type column
    tp_intervals = pd.DataFrame(tp_intervals, columns=["start", "end"])
    tp_intervals["match_type"] = "tp"
    fp_intervals = pd.DataFrame(fp_intervals, columns=["start", "end"])
    fp_intervals["match_type"] = "fp"
    fn_intervals = pd.DataFrame(fn_intervals, columns=["start", "end"])
    fn_intervals["match_type"] = "fn"

    categorized_intervals = pd.concat([tp_intervals, fp_intervals, fn_intervals], ignore_index=True)
    categorized_intervals = categorized_intervals.sort_values(by=["start", "end"], ignore_index=True)
    return categorized_intervals


def _check_input_sanity(
    gsd_list_detected: pd.DataFrame, gsd_list_reference: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # check if input is a dataframe with two columns
    if not isinstance(gsd_list_detected, pd.DataFrame) or not isinstance(gsd_list_reference, pd.DataFrame):
        raise TypeError("`gsd_list_detected` and `gsd_list_reference` must be of type `pandas.DataFrame`.")
    # check if start and end columns are present
    try:
        detected, reference = gsd_list_detected[["start", "end"]], gsd_list_reference[["start", "end"]]
    except KeyError as e:
        raise ValueError(
            "`gsd_list_detected` and `gsd_list_reference` must have columns named 'start' and 'end'."
        ) from e
    return detected, reference


def _get_false_matches_from_overlap_data(overlaps: list[Interval], interval: Interval) -> list[list[int]]:
    f_intervals = []
    for i, overlap in enumerate(overlaps):
        prev_el = overlaps[i - 1] if i > 0 else None
        next_el = overlaps[i + 1] if i < len(overlaps) - 1 else None

        # check if there are false matches before the overlap
        if interval.begin < overlap.begin:
            fn_start = interval.begin
            # check if interval is already covered by a previous overlap
            if prev_el and interval.begin < prev_el.end:
                fn_start = prev_el.end
            f_intervals.append([fn_start, overlap.begin])

        # check if there are false matches after the overlap
        if interval.end > overlap.end:
            fn_end = interval.end
            # check if interval is already covered by a succeeding overlap
            if next_el and interval.end > next_el.begin:
                # skip because this will be handled by the next iteration
                continue
                # fn_end = next_el.begin
            f_intervals.append([overlap.end, fn_end])

    return f_intervals


def find_matches_with_min_overlap(
    gsd_list_detected: pd.DataFrame, gsd_list_reference: pd.DataFrame, overlap_threshold: float = 0.8
) -> pd.DataFrame:
    """
    Find all matches of `gsd_list_detected` in `gsd_list_reference` with at least overlap_threshold overlap.

    The detected and reference dataframes are expected to have columns namend "start" and "end" containing the
    start and end indices of the respective gait sequences.
    As index, a range index or a column of unique identifiers for each gait sequence can be provided.

    Note, that the threshold is enforced in both directions. That means, that the relative overlap of the detected gait
    sequence with respect to the overall length of the detected interval AND to the overall length of the matched
    reference interval must be at least `overlap_threshold`.

    Note, we assume that `gsd_list_detected` has no overlaps, but we don't enforce it!

    Parameters
    ----------
    gsd_list_detected: pd.DataFrame
       Each row contains a detected gait sequence interval as output from the GSD algorithms.
       The respective start index is stored in the first and the stop index in the second column.
       Furthermore, the id of the respective gait sequence can be provided in the third column.
    gsd_list_reference: pd.DataFrame
       Gold standard to validate the detected gait sequences against.
       Should have the same format as `gsd_list_detected`.
    overlap_threshold: float
        The minimum relative overlap between a detected sequence and its reference with respect to the length of both
         intervals.
        Must be larger than 0.5 and smaller than or equal to 1.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the intervals from `gsd_list_detected` that overlap with `gsd_list_reference` with the
        specified minimum overlap.
        The dataframe contains the gait sequence ids as index column as well as the start and end indices of the
        intervals.

    Examples
    --------
    >>> from gaitlink.gsd.validation import find_matches_with_min_overlap
    >>> detected = pd.DataFrame([[0, 10, 0], [20, 30, 1]], columns=["start", "end", "id"]).set_index("id")
    >>> reference = pd.DataFrame([[0, 10, 0], [15, 25, 1]], columns=["start", "end", "id"]).set_index("id")
    >>> result = find_matches_with_min_overlap(detected, reference)
       start  end
    id
    0      0   10
    """
    detected, reference = _check_input_sanity(gsd_list_detected, gsd_list_reference)

    # check if index is unique
    if not gsd_list_detected.index.is_unique:
        raise ValueError("`gsd_list_detected` must have a unique index.")

    if overlap_threshold <= 0.5:
        raise ValueError(
            "overlap_threshold must be greater than 0.5."
            "Otherwise multiple matches between intervals "
            "are possible."
        )
    if overlap_threshold > 1:
        raise ValueError("overlap_threshold must be less than 1." "Otherwise no matches can be returned.")

    tree = IntervalTree.from_tuples(detected.reset_index(names="id")[["start", "end", "id"]].to_numpy())
    final_matches = []

    for _, interval in reference[["start", "end"]].iterrows():
        matches = tree[interval["start"] : interval["end"]]
        if len(matches) > 0:
            for match in matches:
                # First calculate the absolute overlap
                absolute_overlap = match.overlap_size(interval["start"], interval["end"])
                # Then calculate the relative overlap
                relative_overlap_interval = absolute_overlap / (interval["end"] - interval["start"])
                relative_overlap_match = absolute_overlap / (match[1] - match[0])
                if relative_overlap_interval >= overlap_threshold and relative_overlap_match >= overlap_threshold:
                    final_matches.append(list(match)[:3])
                    break

    final_matches = pd.DataFrame(final_matches, columns=["start", "end", "id"]).set_index("id")
    return final_matches


def _get_tn_intervals(categorized_intervals: pd.DataFrame, n_samples: Union[int, None]) -> pd.DataFrame:
    """Add true negative intervals to the categorized intervals by inferring them from the other intervals."""
    if n_samples is None:
        return categorized_intervals

    # add tn intervals
    tn_intervals = []
    for i, (start, _) in enumerate(categorized_intervals[["start", "end"]].itertuples(index=False)):
        if i == 0:
            if start > 0:
                tn_intervals.append([0, start])
        elif start > categorized_intervals.iloc[i - 1]["end"]:
            tn_intervals.append([categorized_intervals.iloc[i - 1]["end"], start])

    if categorized_intervals.iloc[-1]["end"] < n_samples - 1:
        tn_intervals.append([categorized_intervals.iloc[-1]["end"], n_samples - 1])

    tn_intervals = pd.DataFrame(tn_intervals, columns=["start", "end"])
    tn_intervals["match_type"] = "tn"
    return tn_intervals


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
