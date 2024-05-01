"""Class to validate gait sequence detection results."""

import warnings
from collections.abc import Sequence
from itertools import product
from types import FunctionType
from typing import Any, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from intervaltree import IntervalTree
from intervaltree.interval import Interval
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing_extensions import Unpack

from mobgap.utils.evaluation import (
    accuracy_score,
    count_samples_in_intervals,
    count_samples_in_match_intervals,
    npv_score,
    precision_recall_f1_score,
    specificity_score,
)


def calculate_matched_gsd_performance_metrics(
    matches: pd.DataFrame, *, zero_division: Literal["warn", 0, 1] = "warn"
) -> dict[str, Union[float, int]]:
    """
    Calculate commonly known performance metrics for based on the matched overlap with the reference.

    This method assumes that you already calculated the overlapping regions between the ground truth and the detected
    gait sequences using the :func:`~mobgap.gsd.evaluation.categorize_intervals` method.
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

    See the documentation of :func:`~mobgap.utils.evaluation.precision_recall_f1_score` for more details.

    Further metrics are calculated if `matches` contains true negative intervals.
    This can be achieved by passing `n_overall_samples` as additional information to
    :func:`~mobgap.gsd.evaluation.categorize_intervals`.

    - `tn_samples`: Number of samples that are correctly not detected as gait sequences.
    - `specificity`: Specificity of the detected gait sequences.
    - `accuracy`: Accuracy of the detected gait sequences.
    - `npv`: Negative predictive value of the detected gait sequences.

    See the documentation of :func:`~mobgap.utils.evaluation.specificity_score`,
     :func:`~mobgap.utils.evaluation.accuracy_score`, and :func:`~mobgap.utils.evaluation.npv_score` for more
     details.

    Parameters
    ----------
    matches: pd.DataFrame
        A DataFrame as returned by :func:`~mobgap.gsd.evaluation.categorize_intervals`.
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
    categorize_intervals
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

    gsd_metrics = {"tp_samples": tp_samples, "fp_samples": fp_samples, "fn_samples": fn_samples, **precision_recall_f1}

    # tn-dependent metrics
    if tn_samples != 0:
        gsd_metrics["tn_samples"] = tn_samples
        gsd_metrics["specificity"] = specificity_score(matches, zero_division=zero_division)
        gsd_metrics["accuracy"] = accuracy_score(matches, zero_division=zero_division)
        gsd_metrics["npv"] = npv_score(matches, zero_division=zero_division)

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
    categorize_intervals
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

    return gsd_metrics


def categorize_intervals(
    *, gsd_list_detected: pd.DataFrame, gsd_list_reference: pd.DataFrame, n_overall_samples: Optional[int] = None
) -> pd.DataFrame:
    """
    Evaluate detected gait sequence intervals against a reference on a sample-wise level.

    The detected and reference dataframes are expected to have columns namend "start" and "end" containing the
    start and end indices of the respective gait sequences.
    Each sample from the detected interval list is categorized as true positive (tp), false positive (fp),
    false negative (fn), or - if the total length of the recording (``n_overall_samples``) is provided - true negative
    (tn).
    The results are concatenated into intervals of tp, fp, fn, and tn matches and returned as a DataFrame.

    The output of this method can be used to calculate performance metrics using the
    :func:`~mobgap.gsd.evaluation.calculate_matched_gsd_performance_metrics` method.

    Parameters
    ----------
    gsd_list_detected
       Each row contains a detected gait sequence interval as output from the GSD algorithms.
       The respective start index is stored in a column named `start` and the stop index in a column named `end`.
    gsd_list_reference
       Gold standard to validate the detected gait sequences against.
       Should have the same format as `gsd_list_detected`.
    n_overall_samples
        Number of samples in the analyzed recording. If provided, true negative intervals will be added to the result.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the categorized intervals with their `start` and `end` index and the respective
        `match_type`.
        Keep in mind that the intervals are not identical to the intervals in `gsd_list_detected`, but are rather split
        into subsequences according to their match type with the reference.

    Examples
    --------
    >>> from mobgap.gsd.evaluation import categorize_intervals
    >>> detected = pd.DataFrame([[0, 10], [20, 30]], columns=["start", "end"])
    >>> reference = pd.DataFrame([[0, 10], [15, 25]], columns=["start", "end"])
    >>> result = categorize_intervals(detected, reference)
    >>> result.tp_intervals
           start  end match_type
    0      0   10         tp
    1     15   20         fn
    2     20   25         tp
    3     25   30         fp

    See Also
    --------
    calculate_matched_gsd_performance_metrics
        For calculating performance metrics based on the matches returned by this function.
    calculate_unmatched_gsd_performance_metrics
        For calculating performance metrics without matching the detected and reference gait sequences.

    """
    detected, reference = _check_input_sanity(gsd_list_detected, gsd_list_reference)

    if n_overall_samples and n_overall_samples < max(gsd_list_reference["end"].max(), gsd_list_detected["end"].max()):
        raise ValueError(
            "The provided `n_samples` parameter is implausible. The number of samples must be larger than the highest "
            "end value in the detected and reference gait sequences."
        )

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

    # add tn intervals
    if n_overall_samples is not None:
        tn_intervals = _get_tn_intervals(categorized_intervals, n_overall_samples=n_overall_samples)
        categorized_intervals = pd.concat([categorized_intervals, tn_intervals], ignore_index=True)
        categorized_intervals = categorized_intervals.sort_values(by=["start", "end"], ignore_index=True)

    return categorized_intervals


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


def _check_input_sanity(
    gsd_list_detected: pd.DataFrame, gsd_list_reference: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # check if inputs are dataframes
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


def _get_tn_intervals(categorized_intervals: pd.DataFrame, n_overall_samples: Union[int, None]) -> pd.DataFrame:
    """Add true negative intervals to the categorized intervals by inferring them from the other intervals.

    This function requires sorted and non-overlapping intervals in `categorized_intervals`.
    If `n_overall_samples` is not provided, an empty DataFrame is returned.
    """
    if n_overall_samples is None:
        return pd.DataFrame(columns=["start", "end", "match_type"])

    # add tn intervals
    tn_intervals = []
    for i, (start, _) in enumerate(categorized_intervals[["start", "end"]].itertuples(index=False)):
        if i == 0:
            if start > 0:
                tn_intervals.append([0, start])
        elif start > categorized_intervals.iloc[i - 1]["end"]:
            tn_intervals.append([categorized_intervals.iloc[i - 1]["end"], start])

    if categorized_intervals.iloc[-1]["end"] < n_overall_samples - 1:
        tn_intervals.append([categorized_intervals.iloc[-1]["end"], n_overall_samples - 1])

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


def _sanitize_index(ic_list: pd.DataFrame, list_type: Literal["detected", "reference"]) -> tuple[pd.DataFrame, bool]:
    is_multindex = False
    # check if index is a multiindex and raise warning if it is
    if isinstance(ic_list.index, pd.MultiIndex):
        is_multindex = True
        ic_list.index = ic_list.index.to_flat_index()
        ic_list.index.name = f"gs_id_{list_type}"
    # check if indices are unique
    if not ic_list.index.is_unique:
        raise ValueError(f"The index of `gs_list_{list_type}` must be unique!")
    return ic_list, is_multindex


def _plot_intervals_from_df(df: pd.DataFrame, y: int, ax: Axes, **kwargs: Unpack[dict[str, Any]]) -> None:
    label_set = False
    for _, row in df.iterrows():
        label = kwargs.pop("label", None)
        if label and not label_set:
            ax.hlines(y, row["start"], row["end"], lw=20, label=label, **kwargs)
            label_set = True
        else:
            ax.hlines(y, row["start"], row["end"], lw=20, **kwargs)


def categorize_matches_with_min_overlap(
    *,
    gsd_list_detected: pd.DataFrame,
    gsd_list_reference: pd.DataFrame,
    overlap_threshold: float = 0.8,
    multiindex_warning: bool = True,
) -> pd.DataFrame:
    """Evaluate a gait sequence list against a reference sequence-by-sequence with a minimum overlap threshold.

    This compares a gait sequence list against a reference list and classifies each detected sequence as true positive,
    false positive, or false negative.
    A gait sequence is classified as true positive when having at least ``overlap_threshold`` overlap with a reference
    sequence. If a detected sequence has no overlap with any reference sequence, it is classified as false positive.
    If a reference sequence has no overlap with any detected sequence, it is classified as false negative.

    Note, that the threshold is enforced in both directions. That means, that the relative overlap of the detected gait
    sequence with respect to the overall length of the detected interval AND to the overall length of the matched
    reference interval must be at least `overlap_threshold`.

    The detected and reference dataframes are expected to have columns namend "start" and "end" containing the
    start and end indices of the respective gait sequences.
    As index, we support either a single or a multiindex without duplicates
    (i.e., the index must identify each gait sequence uniquely).
    If a multiindex is provided, the single index levels will be ignored for the comparison and matches across different
    index groups will be possible.
    If this is not the intended use case, consider grouping your input data before calling this function
    (see :func:`~mobgap.utils.array_handling.create_multi_groupby`).

    Note, we assume that ``gsd_list_detected`` has no overlaps, but we don't enforce it!

    Parameters
    ----------
    gsd_list_detected: pd.DataFrame
       Each row contains a detected gait sequence interval as output from the GSD algorithms.
       The respective start index is stored in a column named `start` and the stop index in a column named `stop`.
    gsd_list_reference: pd.DataFrame
       Gold standard to validate the detected gait sequences against.
       Should have the same format as `gsd_list_detected`.
    overlap_threshold: float
        The minimum relative overlap between a detected sequence and its reference with respect to the length of both
        intervals.
        Must be larger than 0.5 and smaller than or equal to 1.
    multiindex_warning
        If True, a warning will be raised if the index of the input data is a MultiIndex, explaining that the index
        levels will be ignored for the matching process.
        This exists, as this is a common source of error, when this function is used together with a typical pipeline
        that iterates over individual gait sequences during the processing using :class:`~mobgap.pipeline.GsIterator`.
        Only set this to False, once you understand the two different usecases.

    Returns
    -------
    matches: pandas.DataFrame
        A 3 column dataframe with the column names `gsd_id_detected`,
        `gsd_id_reference`, and `match_type`.
        Each row is a match containing the index value of the detected and the reference list, that belong together,
        or a tuple of index values in case of a multiindex input.
        The `match_type` column indicates the type of match.
        For all gait sequences that have a match in the reference list, this will be "tp" (true positive).
        Gait sequences that do not have a match will be mapped to a NaN and the match-type will be "fp" (false
        positive).
        All reference gait sequences that do not have a counterpart in the detected list
        are marked as "fn" (false negative).

    Examples
    --------
    >>> from mobgap.gsd.evaluation import categorize_matches_with_min_overlap
    >>> detected = pd.DataFrame([[0, 10, 0], [20, 30, 1]], columns=["start", "end", "id"]).set_index("id")
    >>> reference = pd.DataFrame([[0, 10, 0], [15, 25, 1]], columns=["start", "end", "id"]).set_index("id")
    >>> result = categorize_matches_with_min_overlap(detected, reference)
       gsd_id_detected  gs_id_reference match_type
    0               0               0         tp
    1               1               NaN       fp
    2               NaN             1         fn
    """
    detected, reference = _check_input_sanity(gsd_list_detected, gsd_list_reference)
    detected, is_multindex_detected = _sanitize_index(detected, "detected")
    reference, is_multindex_reference = _sanitize_index(reference, "reference")

    if multiindex_warning and (is_multindex_detected or is_multindex_reference):
        warnings.warn(
            "The index of `gsd_list_detected` or `gsd_list_reference` is a MultiIndex. "
            "Please be aware that the index levels will not be regarded separately for the matching process, "
            "and gait sequences might be matched across different index groups, such as recording sessions or "
            "participants.\n"
            "If this is not the intended use case for you, consider grouping your input data before calling the "
            "evaluation function.\n\n"
            "This can be done using the `create_multi_groupby` function from the `mobgap.utils.array_handling`. ",
            stacklevel=1,
        )

    if overlap_threshold <= 0.5:
        raise ValueError(
            "overlap_threshold must be greater than 0.5."
            "Otherwise multiple matches between intervals "
            "are possible."
        )
    if overlap_threshold > 1:
        raise ValueError("overlap_threshold must be less than 1." "Otherwise no matches can be returned.")

    detected["range_index"] = range(len(detected))
    tree = IntervalTree.from_tuples(detected[["start", "end", "range_index"]].to_numpy())
    detected_index = []
    reference_index = []

    for ref_id, interval in reference.reset_index()[["start", "end"]].iterrows():
        matches = tree[interval["start"] : interval["end"]]
        if len(matches) > 0:
            for match in matches:
                det_idx = list(match)[2]
                # First calculate the absolute overlap
                absolute_overlap = match.overlap_size(interval["start"], interval["end"])
                # Then calculate the relative overlap
                relative_overlap_interval = absolute_overlap / (interval["end"] - interval["start"])
                relative_overlap_match = absolute_overlap / (match[1] - match[0])
                if relative_overlap_interval >= overlap_threshold and relative_overlap_match >= overlap_threshold:
                    detected_index.append(det_idx)
                    reference_index.append(ref_id)
                    break

    detected_index_name = "gs_id_detected"
    reference_index_name = "gs_id_reference"

    matches_detected = pd.DataFrame(index=detected.index.copy(), columns=[reference_index_name])
    matches_detected.index.name = detected_index_name

    matches_reference = pd.DataFrame(index=reference.index.copy(), columns=[detected_index_name])
    matches_reference.index.name = reference_index_name

    ic_list_detected_idx = detected.iloc[detected_index].index
    ic_list_reference_idx = reference.iloc[reference_index].index

    matches_detected.loc[ic_list_detected_idx, reference_index_name] = ic_list_reference_idx
    matches_reference.loc[ic_list_reference_idx, detected_index_name] = ic_list_detected_idx

    matches_detected = matches_detected.reset_index()
    matches_reference = matches_reference.reset_index()

    matches = (
        pd.concat([matches_detected, matches_reference])
        .drop_duplicates()
        .sort_values([detected_index_name, reference_index_name])
        .reset_index(drop=True)
    )

    matches.loc[~matches.isna().any(axis=1), "match_type"] = "tp"
    matches.loc[matches[reference_index_name].isna(), "match_type"] = "fp"
    matches.loc[matches[detected_index_name].isna(), "match_type"] = "fn"

    return matches


def combine_det_with_ref_without_matching(
    *, metrics_detected: pd.DataFrame, metrics_reference: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine the detected and reference gait sequence metrics to one dataframe row-by-row based on their index.

    This function is useful when the metrics you are working with are already aggregated on a higher level,
    e.g., per participant, recording date, or trial.
    When you're metrics are calculated on a gait sequence level,
    refer to ~func:`~mobgap.gsd.evaluation.get_matching_gs` instead.

    Parameters
    ----------
    metrics_detected
       Each row corresponds is characterized by a unique index and contains metrics estimated over
       a group of walking bouts.
       The columns contain the calculated metrics that should be compared to the reference metrics.
       The index must be unique and correspond to the index of `metrics_reference`.
    metrics_reference
       Each row corresponds is characterized by a unique index and contains metrics estimated over
       a group of walking bouts.
       The columns contain the metrics serving as reference.
       The index must be unique and correspond to the index of `metrics_detected`.

    Returns
    -------
    combined_metrics: pd.DataFrame
        The detected and reference metrics combined in one DataFrame for all indices present in both input DataFrames.
        The columns are two-level MultiIndex columns, consisting of a `metrics` and an `origin` level.
        As first column level, all columns present in both `metrics_detected` and `metrics_reference` are included.
        The second column level indicates the origin of the respective value, either `detected` or `reference` for
        metrics that were estimated based on the detected or reference gait sequences, respectively.
    """
    if not metrics_detected.index.is_unique:
        raise ValueError("The index of `metrics_detected` must be unique!")
    if not metrics_reference.index.is_unique:
        raise ValueError("The index of `metrics_reference` must be unique!")

    common_indices = metrics_detected.index.intersection(metrics_reference.index)
    if len(common_indices) == 0:
        raise ValueError("No common indices found in `metrics_detected` and `metrics_reference`.")

    detected_metrics = metrics_detected.loc[common_indices]
    reference_metrics = metrics_reference.loc[common_indices]

    combined_metrics = _combine_detected_and_reference_metrics(detected_metrics, reference_metrics)
    return combined_metrics


def get_matching_gs(
    *, metrics_detected: pd.DataFrame, metrics_reference: pd.DataFrame, matches: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract the detected and reference gait sequences that are considered as matches sequence-by-sequence.

    Additionally, the metrics of the detected and reference gait sequences are extracted and returned in a DataFrame
    for further comparison.
    When your metrics are already aggregated on a higher level, such as daily, participant-wise, or session-wise,
    refer to ~func:`~mobgap.gsd.evaluation.combine_det_with_ref_without_matching` instead.

    Parameters
    ----------
    metrics_detected
       Each row corresponds to a detected gait sequence interval as output from the GSD algorithms.
       The columns contain the metrics estimated for each respective gait sequence based on these detected intervals.
       The columns present in both `metrics_detected` and `metrics_reference` are regarded for the matching,
       while the other columns are discarded.
    metrics_reference
       Each row corresponds to a reference gait sequence interval as retrieved from the reference system.
       The columns contain the metrics estimated for each respective gait sequence based on these reference intervals.
       The columns present in both `metrics_detected` and `metrics_reference` are regarded for the matching,
       while the other columns are discarded.
    matches
        A DataFrame containing the matched gait sequences
        as output by :func:`~mobgap.gsd.evaluation.find_matches_with_min_overlap`.
        Must have been calculated based on the same interval data as `metrics_detected` and `metrics_reference`.
        Expected to have the columns `gs_id_detected`, `gs_id_reference`, and `match_type`.

    Returns
    -------
    matches: pd.DataFrame
        The detected gait sequences that are considered as matches assigned to the reference sequences
        they are matching with.
        As index, the unique identifier for each matched gait sequence assigned in the `matches` DataFrame is used.
        The columns are two-level MultiIndex columns, consisting of a `metrics` and an `origin` level.
        As first column level, all columns present in both `metrics_detected` and `metrics_reference` are included.
        The second column level indicates the origin of the respective value, either `detected` or `reference` for
        metrics that were estimated based on the detected or reference gait sequences, respectively.

    Examples
    --------
    >>> from mobgap.gsd.evaluation import categorize_matches_with_min_overlap
    >>> detected = pd.DataFrame([[0, 10, 0], [20, 30, 1]], columns=["start", "end", "id"]).set_index("id")
    >>> reference = pd.DataFrame([[0, 10, 0], [21, 29, 1]], columns=["start", "end", "id"]).set_index("id")
    >>> detected_metrics = pd.DataFrame([[1, 2, 0], [1, 2, 1]], columns=["metric_1", "metric_2", "id"]).set_index("id")
    >>> reference_metrics = pd.DataFrame([[2, 3, 0], [2, 3, 1]], columns=["metric_1", "metric_2", "id"]).set_index("id")
    >>> matches = categorize_matches_with_min_overlap(gsd_list_detected=detected, gsd_list_reference=reference)
    >>> matched_gs = get_matching_gs(
    ...     metrics_detected=detected_metrics, metrics_reference=reference_metrics, matches=matches
    ... )
        metric metric_1           metric_2
        origin detected reference detected reference
    id
    0             1         2        2         3
    1             1         2        2         3

    """
    matches = _check_gs_level_matches_sanity(matches)

    tp_matches = matches.query("match_type == 'tp'")

    detected_matches = _extract_tp_matches(metrics_detected, tp_matches["gs_id_detected"])
    reference_matches = _extract_tp_matches(metrics_reference, tp_matches["gs_id_reference"])

    matches = _combine_detected_and_reference_metrics(detected_matches, reference_matches)
    return matches


def _check_gs_level_matches_sanity(matches: pd.DataFrame) -> pd.DataFrame:
    # check if input is a dataframe
    if not isinstance(matches, pd.DataFrame):
        raise TypeError("`matches` must be of type `pandas.DataFrame`.")
    # check for correct columns
    try:
        matches = matches[["gs_id_detected", "gs_id_reference", "match_type"]]
    except KeyError as e:
        raise ValueError(
            "`matches` must have columns named `gs_id_detected`, `gs_id_reference`, and `match_type`."
        ) from e
    # check if `match_type` column contains only valid values
    if not matches["match_type"].isin(["tp", "fp", "fn"]).all():
        raise ValueError("`match_type` must contain only the values 'tp', 'fp', and 'fn'.")
    return matches


def _extract_tp_matches(metrics: pd.DataFrame, match_indices: pd.Series) -> pd.DataFrame:
    try:
        matches = metrics.loc[match_indices]
    except KeyError as e:
        raise ValueError(
            "The indices from the provided `matches` DataFrame do not fit to the metrics DataFrames. "
            "Please ensure that the `matches` DataFrame is calculated based on the same data "
            "as the `metrics` DataFrames and thus refers to valid indices."
        ) from e
    return matches


def _combine_detected_and_reference_metrics(detected: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    common_columns = list(set(reference.columns).intersection(detected.columns))
    if len(common_columns) == 0:
        raise ValueError("No common columns found in `metrics_detected` and `metrics_reference`.")

    detected = detected[common_columns]
    reference = reference[common_columns]

    matches = detected.merge(reference, left_index=True, right_index=True, suffixes=("_det", "_ref"))

    # construct MultiIndex columns
    matches.columns = pd.MultiIndex.from_product(
        [["detected", "reference"], common_columns], names=["origin", "metric"]
    )
    # make 'metrics' level the uppermost level and sort columns accordingly for readability
    matches = matches.swaplevel(axis=1).sort_index(axis=1, level=0)
    return matches


def assign_error_metrics(
    df: pd.DataFrame,
    parameters: Union[str, list[str]],
    zero_division_hint: Union[Literal["warn", "raise"], float] = "warn",
) -> pd.DataFrame:
    """Derive error metrics from given columns of a DataFrame and add them to the DataFrame.

    The error metrics that are calculated for each parameter are:
    - error: The difference between the detected and reference value.
    - rel_error: The relative error, calculated as the error divided by the reference value.
    - abs_rel_error: The absolute value of the relative error.
    - abs_error: The absolute value of the error.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the columns to derive the error metrics from.
        It is retrieved from one of the functions ~func:`~mobgap.gsd.evaluation.combine_det_with_ref_without_matching`
        or ~func:`~mobgap.gsd.evaluation.get_matching_gs`.
        Needs to have a MultiIndex column structure with the first level being the metric name and the second level
        being the origin of the metric (e.g., "detected" or "reference").
    parameters : Union[str, list[str]]
        The name of the gait parameters or a list of parameter names for which the error metrics should be calculated.
        Each of the specified parameters must be present in `df` together with a reference and a detected value.
    zero_division_hint : "warn", "raise" or np.nan, default="warn"
        Controls the behavior when there is a zero division. If set to "warn",
        affected metrics are set to NaN and a warning is raised.
        If set to "raise", a ZeroDivisionError is raised.
        If set to `np.nan`, the warning is suppressed and the affected metrics are set to NaN.
        Zero division can occur if the reference value is zero for a given parameter, i.e., n_turns = 0

    Returns
    -------
    df : pd.DataFrame
        The input DataFrame with the error metrics for the specified parameters added in-place.
    """
    if isinstance(parameters, str):
        parameters = [parameters]
    for value in parameters:
        if value not in df.columns.get_level_values("metric"):
            raise ValueError(f"Column '{value}' not found in DataFrame.")
        try:
            detected_value = df[value]["detected"]
            reference_value = df[value]["reference"]
        except KeyError as e:
            raise ValueError(f"Column '{value}' does not contain both 'detected' and 'reference' values.") from e

        # explicitly calculate the insertion order to keep columns in a logical order
        insert_index = df.columns.get_loc(value).stop
        df.insert(insert_index, (value, "error"), detected_value - reference_value)
        df.insert(
            insert_index + 1, (value, "rel_error"), (df[value]["error"] / reference_value).replace(np.inf, np.nan)
        )
        df.insert(insert_index + 2, (value, "abs_rel_error"), df[value]["rel_error"].abs())
        df.insert(insert_index + 3, (value, "abs_error"), df[value]["error"].abs())

        # handle zero division
        if (reference_value == 0).any():
            if zero_division_hint not in ["warn", "raise", np.nan]:
                raise ValueError('"zero_division" must be set to "warn", "raise" or `np.nan`!')
            if zero_division_hint == "raise":
                raise ZeroDivisionError(f"Zero division occurred because reference values of {value} contain zeroes.")
            if zero_division_hint == "warn":
                warnings.warn(
                    f"Zero division occurred because because reference values of {value} contain zeroes. "
                    "Affected error metrics are set to NaN.",
                    UserWarning,
                    stacklevel=2,
                )
    return df


# TODO: how to handle icc?
"""
def icc(matches_df: pd.DataFrame, formatter:str="{icc:#.2f} {_a}[{q05:.2f}, {q95:.2f}]") -> str:
    Calculate the ICC.

    We expect the df to the "AverageStrideSpeed_ref", "AverageStrideSpeed_single" columns.

    only_speed = matches_df.loc[:, ["AverageStrideSpeed_ref", "AverageStrideSpeed_single"]].rename(
        columns=lambda c: tuple(c.split("_"))
    )
    only_speed.columns = pd.MultiIndex.from_tuples(only_speed.columns, names=("measure", "system"))
    only_speed = (
        only_speed.assign(wb_uid=lambda df_: pd.factorize(df_.index)[0])
        .set_index("wb_uid", append=True)
        .stack("system")
        .reset_index(["system", "wb_uid"])
        .astype({"AverageStrideSpeed": float})
    )
    icc, ci95 = intraclass_corr(only_speed, ratings="AverageStrideSpeed", raters="system", targets="wb_uid").loc[
        0, ["ICC", "CI95%"]
    ]
    return formatter.format(icc=icc, _a=NO_BREAK_SPACE, **dict(zip(("q05", "q95"), ci95)))
    """


def quantiles(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> tuple[float, float]:
    """Calculate the quantiles of a measure."""
    return tuple(series.quantile([lower, upper]).to_numpy())


def loa(series: pd.Series, agreement: float = 1.96) -> tuple[float, float]:
    """Calculate the limits of agreement of a measure."""
    mean = series.mean()
    std = series.std()
    return mean - std * agreement, mean + std * agreement


def mdc(error: pd.Series) -> float:
    # TODO: isn't this supposed to be a tuple then?
    # Is this metric even required?
    """Calculate the mean difference and the confidence interval of a measure."""
    return 1.96 * np.sqrt(2) * error.std()


def get_aggregator(
    aggregate: Union[str, Sequence[str]],
    metric: Union[str, Sequence[str]],
    origin: Union[str, Sequence[str]] = ("detected", "reference"),
) -> dict[tuple[str], list[Union[str, FunctionType]]]:
    """Create a dictionary for collecting aggregated metrics.

    Contains the aggregation functions for a set of aggregations, metrics to aggregate,
    and origins to extract the metrics from. The return value can directly be passed to
    ~func:`~mobgap.gsd.evaluation.apply_aggregations` as the `aggregate_funcs` parameter.

    Parameters
    ----------
    aggregate : Union[str, Sequence[str]]
        The aggregation functions to apply to the metrics, e.g. "mean" (mean), "std" (standard deviation),
        "quantiles" (tuple of 5th and 95th percent quantile), "loa" (tuple of limits of agreement),
        or "mdc" (mean difference).
    metric : Union[str, Sequence[str]]
        The metrics for which the aggregates should be applied.
    origin : Union[str, Sequence[str]], optional
        The origin of the data to aggregate over, e.g. "detected" or "reference" to directly aggregate over the data,
        or "error", "rel_error", "abs_error", or "abs_rel_error" to aggregate over the errors between
        detected and reference.

    Returns
    -------
    aggregator : dict[tuple[str], list[Union[str, FunctionType]]]:
        A dictionary containing the aggregation functions for each metric and origin combination in the form of
        {("metric", "origin"): [aggregation_function, ...], ...}.

    Note
    ----
    If you want to apply different aggregations to different metrics or origins,
    you can call this function multiple times and merge the results.
    However, be aware that the aggregations will be overwritten if the same metric and origin combinations
    are used in multiple calls.
    """
    if isinstance(aggregate, str):
        aggregate = [aggregate]
    aggregate = list(set(aggregate))
    if isinstance(metric, str):
        metric = [metric]
    metric = list(set(metric))
    if isinstance(origin, str):
        origin = [origin]
    origin = list(set(origin))

    # check if all aggregate strings are valid functions
    aggregate_funcs = []
    for agg in aggregate:
        aggregate_funcs.append(_sanitize_agg(agg))

    aggregator = {}
    metric_origin = list(product(metric, origin))
    for metric_origin_comb in metric_origin:
        aggregator[metric_origin_comb] = aggregate_funcs
    return aggregator


def get_default_aggregator() -> dict[tuple[str], list[Union[str, FunctionType]]]:
    """
    Return a dictionary containing all important aggregations utilized in Mobilise-D.

    This dictionary can directly be passed to ~func:`~mobgap.gsd.evaluation.apply_aggregations` as the `aggregate_funcs`
    parameter to calculate the desired metrics.
    """
    # TODO: are these all required parameters?

    parameter_columns = [
        "cadence_spm",
        "duration_s",
        "n_steps",
        "n_turns",
        "stride_duration_s",
        "stride_length_m",
        "walking_speed_mps",
    ]

    default_agg = {
        **get_aggregator(aggregate=["quantiles"], metric=parameter_columns, origin="detected"),
        **get_aggregator(aggregate=["quantiles"], metric=parameter_columns, origin="reference"),
        **get_aggregator(aggregate=["mean", "loa"], metric=parameter_columns, origin="error"),
        **get_aggregator(aggregate=["mean", "loa"], metric=parameter_columns, origin="rel_error"),
        **get_aggregator(aggregate=["quantiles"], metric=parameter_columns, origin="abs_error"),
        **get_aggregator(aggregate=["quantiles"], metric=parameter_columns, origin="abs_rel_error"),
    }

    return default_agg


def apply_aggregations(
    df: pd.DataFrame, aggregate_funcs: dict[tuple[str] : list[Union[str, FunctionType]]]
) -> pd.DataFrame:
    """Apply a set of aggregation functions to a metrics DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the metrics to aggregate.
        It is retrieved from one of the functions ~func:`~mobgap.gsd.evaluation.combine_det_with_ref_without_matching`
        or ~func:`~mobgap.gsd.evaluation.get_matching_gs`.
        Needs to have a MultiIndex column structure with the first level being the metric name and the second level
        being the origin of the metric (e.g., "detected" or "reference").
        If error metrics are of interest, they can also be included
        by calling ~func:`~mobgap.gsd.evaluation.assign_error_metrics` beforehand.

    aggregate_funcs : dict[tuple[str], list[Union[str, FunctionType]]]
        A dictionary containing the aggregation functions for each metric and origin combination in the form of
        {("metric", "origin"): [aggregation_function, ...], ...}.
        Can be created using ~func:`~mobgap.gsd.evaluation.get_aggregator`
        or ~func:`~mobgap.gsd.evaluation.get_default_aggregator`.
    """
    # input validation
    _sanitize_combined_metrics(df, aggregate_funcs.keys())
    for key, agg in aggregate_funcs.items():
        aggregate_funcs[key] = [_sanitize_agg(a) for a in agg]

    agg_metrics = df.agg(aggregate_funcs)
    agg_metrics.index.name = "aggregate"
    agg_metrics = agg_metrics.stack(level=("metric", "origin"), future_stack=True).dropna()
    agg_metrics = agg_metrics.reorder_levels(["metric", "origin", "aggregate"]).sort_index(level=0)
    return agg_metrics


def _sanitize_combined_metrics(df: pd.DataFrame, metric_origin_combinations: list[tuple[str]]) -> None:
    # check if input is a dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be of type `pandas.DataFrame`.")
    # check if input has the correct MultiIndex structure
    if not all([isinstance(df.columns, pd.MultiIndex), df.columns.names == ["metric", "origin"]]):
        raise ValueError("`df` must have a MultiIndex column structure.")
    # check if all metric_origin_combinations are valid and present in the input
    for metric_origin_comb in metric_origin_combinations:
        if not isinstance(metric_origin_comb, tuple):
            raise TypeError("All keys in `aggregate_funcs` must be tuples.")
        if metric_origin_comb not in df.columns:
            raise ValueError(f"Column '{metric_origin_comb}' not found in DataFrame.")


def _sanitize_agg(agg: Union[str, FunctionType]) -> Union[FunctionType, str]:
    # agg is function type already
    if isinstance(agg, FunctionType):
        return agg
    # agg refers to custom function
    if agg in globals() and callable(globals()[agg]):
        return globals()[agg]
    # built-in functions
    try:
        callable(getattr(pd.Series(), agg))
    except AttributeError as e:
        raise ValueError(f"Function '{agg}' is not implemented and thus not a valid aggregation function.") from e
    return agg


__all__ = [
    "categorize_intervals",
    "categorize_matches_with_min_overlap",
    "calculate_matched_gsd_performance_metrics",
    "calculate_unmatched_gsd_performance_metrics",
    "plot_categorized_intervals",
    "combine_det_with_ref_without_matching",
    "get_matching_gs",
    "assign_error_metrics",
    "quantiles",
    "loa",
    "mdc",
    "get_aggregator",
    "get_default_aggregator",
    "apply_aggregations",
]

if __name__ == "__main__":
    """
    detected = pd.DataFrame([[0, 10, 0], [11, 12, 1], [20, 30, 2]], columns=["start", "end", "id"]).set_index("id")
    reference = pd.DataFrame([[0, 10, 0], [13, 14, 1], [21, 29, 2]], columns=["start", "end", "id"]).set_index("id")
    matches = categorize_matches_with_min_overlap(gsd_list_detected=detected, gsd_list_reference=reference)
    detected_metrics = pd.DataFrame(
        [[1, 2, 0], [1, 2, 1], [3, 4, 2]], columns=["metric_1", "metric_2", "id"]
    ).set_index("id")
    matched_gs = get_matching_gs(
        gsd_list_detected=detected_metrics, gsd_list_reference=detected_metrics, matches=matches
    )
    matches_with_err = matched_gs.pipe(lambda df_: assign_error_metrics(df_, ["metric_1", "metric_2"]))
    print(matches_with_err)
    aggregator = get_default_aggregator()
    # TODO: values are overwritten when keys are reused - is this a problem?
    aggregator = {
        **get_aggregator(aggregate=["mean", "std"], metric=["metric_1", "metric_2"], origin="detected"),
        **get_aggregator(aggregate=["loa", "mdc", "quantiles"], metric="metric_2", origin="rel_error"),
    }
    print(aggregator)
    agg_results = matches_with_err.pipe(apply_aggregations, aggregator)
    print(agg_results)
    """
    # construct MultiIndex
    detected = pd.DataFrame([[0, 10, 0, 0], [20, 30, 0, 1]], columns=["start", "end", "id_0", "id_1"]).set_index(
        ["id_0", "id_1"]
    )
    reference = pd.DataFrame([[0, 10, 0, 0], [21, 29, 0, 1]], columns=["start", "end", "id_0", "id_1"]).set_index(
        ["id_0", "id_1"]
    )
    detected_metrics = pd.DataFrame(
        [[1, 2, 0, 0], [1, 2, 0, 1]], columns=["metric_1", "metric_2", "id_0", "id_1"]
    ).set_index(["id_0", "id_1"])
    reference_metrics = pd.DataFrame(
        [[3, 4, 0, 0], [3, 4, 0, 1]], columns=["metric_2", "metric_1", "id_0", "id_1"]
    ).set_index(["id_0", "id_1"])
    matches = categorize_matches_with_min_overlap(gsd_list_detected=detected, gsd_list_reference=reference)
    matched_gs = get_matching_gs(
        metrics_detected=detected_metrics, metrics_reference=reference_metrics, matches=matches
    )
    matches = combine_det_with_ref_without_matching(
        metrics_detected=detected_metrics, metrics_reference=reference_metrics
    )

    # matches_with_err = matches.pipe(lambda df_: assign_error_metrics(df_, ["metric_1", "metric_2"]))
    matches_with_err = assign_error_metrics(matches, ["metric_1", "metric_2"])
    aggregator = get_default_aggregator()
    agg_results = apply_aggregations(matches_with_err, aggregator)
    print(agg_results)
