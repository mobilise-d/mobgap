"""Class to validate gait sequence detection results."""

import warnings
from collections.abc import Hashable, Sequence
from functools import wraps
from typing import Any, Callable, Literal, NamedTuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from intervaltree import IntervalTree
from intervaltree.interval import Interval
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pingouin import intraclass_corr
from typing_extensions import Unpack

from mobgap.utils.evaluation import (
    accuracy_score,
    count_samples_in_intervals,
    count_samples_in_match_intervals,
    npv_score,
    precision_recall_f1_score,
    specificity_score,
)


class CustomOperation(NamedTuple):
    """Metadata for custom aggregations and transformations."""

    identifier: Union[Hashable, Sequence, str]
    function: Union[Callable, list[Callable]]
    column_name: Union[str, tuple[str, ...]]

    @property
    def _TAG(self) -> str:  # noqa: N802
        return "CustomOperation"


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
    Additionally, note that this method won't return any new intervals
    (as done in :func:`~mobgap.gsd.evaluation.categorize_intervals`).
    Instead, the comparison is done on a sequence-by-sequence level based on the provided intervals.

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
    >>> detected = pd.DataFrame(
    ...     [[0, 10, 0], [20, 30, 1]], columns=["start", "end", "id"]
    ... ).set_index("id")
    >>> reference = pd.DataFrame(
    ...     [[0, 10, 0], [15, 25, 1]], columns=["start", "end", "id"]
    ... ).set_index("id")
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

    matches.index.name = "match_id"
    matches.loc[~matches.isna().any(axis=1), "match_type"] = "tp"
    matches.loc[matches[reference_index_name].isna(), "match_type"] = "fp"
    matches.loc[matches[detected_index_name].isna(), "match_type"] = "fn"

    return matches


def get_matching_intervals(
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
    >>> detected = pd.DataFrame(
    ...     [[0, 10, 0], [20, 30, 1]], columns=["start", "end", "id"]
    ... ).set_index("id")
    >>> reference = pd.DataFrame(
    ...     [[0, 10, 0], [21, 29, 1]], columns=["start", "end", "id"]
    ... ).set_index("id")
    >>> detected_metrics = pd.DataFrame(
    ...     [[1, 2, 0], [1, 2, 1]], columns=["metric_1", "metric_2", "id"]
    ... ).set_index("id")
    >>> reference_metrics = pd.DataFrame(
    ...     [[2, 3, 0], [2, 3, 1]], columns=["metric_1", "metric_2", "id"]
    ... ).set_index("id")
    >>> matches = categorize_matches_with_min_overlap(
    ...     gsd_list_detected=detected, gsd_list_reference=reference
    ... )
    >>> matched_gs = get_matching_intervals(
    ...     metrics_detected=detected_metrics,
    ...     metrics_reference=reference_metrics,
    ...     matches=matches,
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

    combined_matches = _combine_detected_and_reference_metrics(
        detected_matches, reference_matches, tp_matches=tp_matches
    )

    return combined_matches


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


def _combine_detected_and_reference_metrics(
    detected: pd.DataFrame, reference: pd.DataFrame, tp_matches: Union[pd.DataFrame, None] = None
) -> pd.DataFrame:
    # if wb_id in index, add it as a column to preserve it in the combined DataFrame
    if "wb_id" in detected.index.names and "wb_id" in reference.index.names:
        detected.insert(0, "wb_id", detected.index.get_level_values("wb_id"))
        reference.insert(0, "wb_id", reference.index.get_level_values("wb_id"))

    common_columns = list(set(reference.columns).intersection(detected.columns))
    if len(common_columns) == 0:
        raise ValueError("No common columns found in `metrics_detected` and `metrics_reference`.")

    detected = detected[common_columns]
    reference = reference[common_columns]

    if tp_matches is not None:
        detected.index = tp_matches.index
        reference.index = tp_matches.index

    matches = detected.merge(reference, left_index=True, right_index=True, suffixes=("_det", "_ref"))

    # construct MultiIndex columns
    matches.columns = pd.MultiIndex.from_product(
        [["detected", "reference"], common_columns]  # , names=["origin", "metric"]
    )
    # make 'metrics' level the uppermost level and sort columns accordingly for readability
    matches = matches.swaplevel(axis=1).sort_index(axis=1, level=0)
    return matches


def _get_data_from_identifier(
    df: pd.DataFrame, identifier: Union[Hashable, Sequence, str], num_levels: Union[int, None] = 1
) -> pd.DataFrame:
    try:
        data = df.loc[:, identifier]
    except KeyError as e:
        raise ValueError(f"Column(s) '{identifier}' not found in DataFrame.") from e
    if num_levels:
        data_num_levels = 1 if isinstance(data, pd.Series) else data.columns.nlevels
        if data_num_levels != num_levels:
            raise ValueError(f"Data selected by '{identifier}' must have {num_levels} level(s).")
    return data


def _handle_zero_division(
    divisor: Union[pd.Series, pd.DataFrame],
    zero_division_hint: Union[Literal["warn", "raise"], float],
    caller_fct_name: str,
) -> None:
    if (divisor == 0).any():
        if zero_division_hint not in ["warn", "raise", np.nan]:
            raise ValueError('"zero_division" must be set to "warn", "raise" or `np.nan`!')
        if zero_division_hint == "raise":
            raise ZeroDivisionError(f"Zero division occurred in {caller_fct_name} because divisor contains zeroes.")
        if zero_division_hint == "warn":
            warnings.warn(
                f"Zero division occurred in {caller_fct_name} because divisor contains zeroes. "
                "Affected error metrics are set to NaN.",
                UserWarning,
                stacklevel=2,
            )


def error(df: pd.DataFrame, reference_col_name: str = "reference", detected_col_name: str = "detected") -> pd.Series:
    """
    Calculate the error between the detected and reference values.

    Parameters
    ----------
    df
        The DataFrame containing the reference and detected values.
    reference_col_name
        The identifier of the column containing the reference values.
    detected_col_name
        The identifier of the column containing the detected values.

    Returns
    -------
    error
        The error between the detected and reference values in the form `detected` - `reference`
    """
    ref, det = _get_data_from_identifier(df, reference_col_name), _get_data_from_identifier(df, detected_col_name)
    return det - ref


def rel_error(
    df: pd.DataFrame,
    reference_col_name: str = "reference",
    detected_col_name: str = "detected",
    zero_division_hint: Union[Literal["warn", "raise"], float] = "warn",
) -> pd.Series:
    """
    Calculate the relative error between the detected and reference values.

    Parameters
    ----------
    df
        The DataFrame containing the reference and detected values.
    reference_col_name
        The identifier of the column containing the reference values.
    detected_col_name
        The identifier of the column containing the detected values.
    zero_division_hint
        How to handle zero division errors. Can be one of "warn" (warning is given, respective values are set to NaN),
        "raise" (error is raised), or "np.nan" (respective values are silently set to NaN).

    Returns
    -------
    rel_error
        The relative error between the detected and reference values
        in the form (`detected` - `reference`) / `reference`.
    """
    ref, det = (
        _get_data_from_identifier(df, reference_col_name),
        _get_data_from_identifier(df, detected_col_name),
    )
    # inform about zero division if it occurs
    _handle_zero_division(ref, zero_division_hint, "rel_error")
    result = (det - ref) / ref
    result = result.replace(np.inf, np.nan)
    return result


def abs_error(
    df: pd.DataFrame, reference_col_name: str = "reference", detected_col_name: str = "detected"
) -> pd.Series:
    """
    Calculate the absolute error between the detected and reference values.

    Parameters
    ----------
    df
        The DataFrame containing the reference and detected values.
    reference_col_name
        The identifier of the column containing the reference values.
    detected_col_name
        The identifier of the column containing the detected values.

    Returns
    -------
    abs_error
        The absolute error between the detected and reference values in the form `abs(detected - reference)`.
    """
    ref, det = _get_data_from_identifier(df, reference_col_name), _get_data_from_identifier(df, detected_col_name)
    return abs(det - ref)


def abs_rel_error(
    df: pd.DataFrame,
    reference_col_name: str = "reference",
    detected_col_name: str = "detected",
    zero_division_hint: Union[Literal["warn", "raise"], float] = "warn",
) -> pd.Series:
    """
    Calculate the absolute relative error between the detected and reference values.

    Parameters
    ----------
    df
        The DataFrame containing the reference and detected values.
    reference_col_name
        The identifier of the column containing the reference values.
    detected_col_name
        The identifier of the column containing the detected values.
    zero_division_hint
        How to handle zero division errors. Can be one of "warn" (warning is given, respective values are set to NaN),
        "raise" (error is raised), or "np.nan" (respective values are silently set to NaN).

    Returns
    -------
    abs_rel_error
        The absolute relative error between the detected and reference values
        in the form `abs((detected - reference) / reference)`.
    """
    ref, det = (
        _get_data_from_identifier(df, reference_col_name),
        _get_data_from_identifier(df, detected_col_name),
    )
    # inform about zero division if it occurs
    _handle_zero_division(ref, zero_division_hint, "abs_rel_error")
    result = abs((det - ref) / ref)
    result = result.replace(np.inf, np.nan)
    return result


def icc(
    df: pd.DataFrame, reference_col_name: str = "reference", detected_col_name: str = "detected"
) -> tuple[float, float]:
    """
    Calculate the intraclass correlation coefficient (ICC) for the detected and reference values.

    Parameters
    ----------
    df
        The DataFrame containing the reference and detected values.
    reference_col_name
        The identifier of the column containing the reference values.
    detected_col_name
        The identifier of the column containing the detected values.

    Returns
    -------
    icc, ci95
        A tuple containing the intraclass correlation coefficient (ICC) as first item
        and the lower and upper bound of its 95% confidence interval (CI95%) as second item.

    """
    df = _get_data_from_identifier(df, [reference_col_name, detected_col_name], num_levels=1)
    df = (
        df.reset_index(drop=True)
        .rename_axis("targets", axis=0)
        .rename_axis("rater", axis=1)
        .stack()
        .rename("value")
        .reset_index()
    )
    icc, ci95 = intraclass_corr(data=df, targets="targets", raters="rater", ratings="value").loc[0, ["ICC", "CI95%"]]
    return icc, ci95


def quantiles(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> tuple[float, float]:
    """Calculate the quantiles of a measure.

    Parameters
    ----------
    series
        The Series containing the data column of interest.
    lower
        The lower quantile to calculate.
    upper
        The upper quantile to calculate.

    Returns
    -------
    quantiles
        The lower and upper quantiles as a tuple.
    """
    return tuple(series.quantile([lower, upper]).to_numpy())


def loa(series: pd.Series, agreement: float = 1.96) -> tuple[float, float]:
    """Calculate the limits of agreement of a measure.

    Parameters
    ----------
    series
        The Series containing the data column of interest.
    agreement
        The agreement level for the limits of agreement.

    Returns
    -------
    loa
        The lower and upper limits of agreement as a tuple.
    """
    mean = series.mean()
    std = series.std()
    return mean - std * agreement, mean + std * agreement


def get_default_error_transformations() -> list[tuple[str, list[callable]]]:
    """
    Get all default error metrics used in Mobilise-D.

    This list can directly be passed to ~func:`~mobgap.gsd.evaluation.apply_transformations` as the `transformations`
    parameter to calculate the desired metrics.
    """
    metrics = [
        "cadence_spm",
        "duration_s",
        "n_steps",
        "n_turns",
        "stride_duration_s",
        "stride_length_m",
        "walking_speed_mps",
    ]
    default_errors = [error, rel_error, abs_error, abs_rel_error]
    error_metrics = [*((m, default_errors) for m in metrics)]
    return error_metrics


def get_default_aggregations() -> (
    list[Union[tuple[tuple[str, ...], Union[list[Union[callable, str]], callable, str]], CustomOperation]]
):
    """
    Return a list containing all important aggregations utilized in Mobilise-D.

    This list can directly be passed to ~func:`~mobgap.gsd.evaluation.apply_aggregations` as the `aggregations`
    parameter to calculate the desired metrics.
    """
    metrics = [
        "cadence_spm",
        "duration_s",
        "n_steps",
        "n_turns",
        "stride_duration_s",
        "stride_length_m",
        "walking_speed_mps",
    ]

    default_agg = [
        *(
            ((m, o), ["mean", quantiles])
            for m in metrics
            for o in ["detected", "reference", "abs_error", "abs_rel_error"]
        ),
        *(((m, o), ["mean", loa]) for m in metrics for o in ["error", "rel_error"]),
        *[CustomOperation(identifier=m, function=icc, column_name=(m, "all")) for m in metrics],
    ]

    return default_agg


def apply_transformations(
    df: pd.DataFrame, transformations: list[Union[tuple[str, Union[callable, list[callable]]], CustomOperation]]
) -> pd.DataFrame:
    """Apply a set of transformations to a DMO DataFrame.

    Returns a DataFrame with one column per transformation.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the metrics to transform.
        It is retrieved from one of the functions ~func:`~mobgap.gsd.evaluation.combine_det_with_ref_without_matching`
        or ~func:`~mobgap.gsd.evaluation.get_matching_intervals`.
        Needs to have a MultiIndex column structure with the first level being the metric name and the second level
        being the origin of the metric (e.g., "detected" or "reference").

    transformations : list[tuple[str, Union[callable, list[callable]], CustomOperation]]

        A list specifying which transformation functions are to be applied for which metrics and data origins.
        There are two ways to define transformations:

        1.  As a tuple in the format `(<metric>, <function>)`,
            where `<metric>` is the metric column to evaluate,
            and `<function>` is the function (or a list of functions) to apply.
            The output dataframe will have a multilevel column consisting of a `metric` level and a `function` level.

        2.  As a named tuple of type `CustomOperation` taking three arguments:
            `identifier`, `function`, and `column_name`.
            `identifier` is a valid loc identifier selecting one or more columns from the dataframe,
            `function` is the (custom) transformation function or list of functions to apply,
            and `column_name` is the name of the resulting column in the output dataframe.
            In case of a single-level output column, `column_name` is a string, whereas for multi-level output columns,
            it is a tuple of strings.
            This allows for more complex transformations that require multiple columns as input.

        The default list of aggregations can be retrieved
        using ~func:`~mobgap.gsd.evaluation.get_default_transformations`.

    """
    transformation_results = []
    column_names = []
    for transformation in transformations:
        if getattr(transformation, "_TAG", None) == "CustomOperation":
            data = _get_data_from_identifier(df, transformation.identifier, num_levels=None)
            result = transformation.function(data)
            transformation_results.append(result)
            column_names.append(transformation.column_name)
        else:
            metric, functions = transformation
            if not isinstance(functions, list):
                functions = [functions]
            data = _get_data_from_identifier(df, metric, num_levels=None)
            for fct in functions:
                try:
                    fct_name = fct.__name__
                except AttributeError as e:
                    raise ValueError(
                        f"Transformation function {fct} for identifier {metric} does not have a `__name__`-Attribute. "
                        "Please use a named function or assign a name."
                    ) from e
                result = fct(data)
                transformation_results.append(result)
                column_names.append((metric, fct_name))
    # combine results
    try:
        transformation_results = pd.concat(transformation_results, axis=1)
    except TypeError as e:
        raise ValueError(
            "The transformation results could not be concatenated. "
            "This is likely due to an unexpected return type of a custom function."
            "Please ensure that the return type is a pandas Series for all custom functions."
        ) from e
    try:
        transformation_results.columns = pd.MultiIndex.from_tuples(column_names)
    except ValueError as e:
        raise ValueError(
            f"The expected number of column names {len(pd.MultiIndex.from_tuples(column_names))} "
            f"does not match with the actual number {transformation_results.shape[1]} of columns "
            "in the transformed DataFrame."
            "This is likely due to an unexpected return shape of a CustomOperation function."
        ) from e
    return transformation_results


def apply_aggregations(
    df: pd.DataFrame,
    aggregations: list[
        Union[
            tuple[Union[str, tuple[str, ...]], Union[Union[callable, str], list[Union[callable, str]]]], CustomOperation
        ]
    ],
) -> pd.Series:
    """Apply a set of aggregations to a DMO DataFrame.

    Returns a Series with one entry per aggregation.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the metrics to aggregate.
        It is retrieved from one of the functions ~func:`~mobgap.gsd.evaluation.combine_det_with_ref_without_matching`
        or ~func:`~mobgap.gsd.evaluation.get_matching_intervals`.
        Needs to have a MultiIndex column structure with the first level being the metric name and the second level
        being the origin of the metric (e.g., "detected" or "reference").
        If further derived metrics, such as error metrics are of interest, they can also be included
        by calling ~func:`~mobgap.gsd.evaluation.apply_transformations` beforehand.

    aggregations : list
        A list specifying which aggregation functions are to be applied for which metrics and data origins.
        There are two ways to define aggregations:

        1.  As a tuple in the format `(<identifier>, <aggregation>)`.
            In this case, the operation is performed based on exactly one column from the input df.
            Therefore, <identifier> can either be a string representing the name of the column to evaluate
            (for data with single-level columns),
            or a tuple of strings uniquely identifying the column to evaluate.
            In case of the standard Mobilise-D data structure, this would be a tuple (<metric>, <origin>),
            where `<metric>` is the metric column to evaluate,
            `<origin>` is the specific column from which data should be utilized
            (e.g., `detected`, `reference`, or `error`).
            Furthermore, `<aggregation>` is the function or the list of functions to apply.
            The output dataframe will have a multilevel column consisting of the `metric` level and the
            `origin` level.

        2.  As a named tuple of type `CustomOperation` taking three arguments:
            `identifier`, `function`, and `column_name`.
            `identifier` is a valid loc identifier selecting one or more columns from the dataframe,
            `function` is the (custom) aggregation function or list of functions to apply,
            and `column_name` is the name of the resulting column in the output dataframe.
            In case of a single-level output column, `column_name` is a string, whereas for multi-level output columns,
            it is a tuple of strings.
            This allows for more complex aggregations that require multiple columns as input,
            e.g., the intraclass correlation coefficient (ICC).

        The default list of aggregations can be retrieved using ~func:`~mobgap.gsd.evaluation.get_default_aggregations`.
    """
    manual_aggregations, agg_aggregations = _collect_manual_and_agg_aggregations(aggregations)

    # apply built-in aggregations
    agg_aggregation_results = []
    for key, aggregation in agg_aggregations.items():
        try:
            aggregation_result = df.agg({key: aggregation})
            agg_aggregation_results.append(
                aggregation_result.stack(level=np.arange(df.columns.nlevels).tolist(), future_stack=True)
            )
        except KeyError as e:
            raise ValueError("Column(s) specified in aggregations not found in DataFrame.") from e
    if agg_aggregation_results:
        agg_aggregation_results = pd.concat(agg_aggregation_results)

    manual_aggregation_results = _apply_manual_aggregations(df, manual_aggregations)

    # if only one type of aggregation was applied, return the result directly
    if manual_aggregations and not agg_aggregations:
        return manual_aggregation_results
    if agg_aggregations and not manual_aggregations:
        return agg_aggregation_results

    # otherwise, concatenate the results
    try:
        _check_number_of_index_levels([agg_aggregation_results, manual_aggregation_results])
    except ValueError as e:
        raise ValueError(
            "The aggregation results from automatic and custom aggregation could not be concatenated. "
            "This is likely caused by an inconsistent number index levels in them."
        ) from e
    aggregation_results = pd.concat([agg_aggregation_results, manual_aggregation_results])

    return aggregation_results


def _collect_manual_and_agg_aggregations(
    aggregations: list[
        Union[
            tuple[Union[str, tuple[str, ...]], Union[Union[callable, str], list[Union[callable, str]]]], CustomOperation
        ]
    ],
) -> tuple[list[CustomOperation], dict[tuple[str, str], list[Union[str, Callable]]]]:
    manual_aggregations = []
    agg_aggregations = {}
    for agg in aggregations:
        if getattr(agg, "_TAG", None) == "CustomOperation":
            manual_aggregations.append(agg)
        else:
            key, aggregation = agg
            if not isinstance(aggregation, list):
                aggregation = [aggregation]
            wrapped_aggregation = []
            for fct in aggregation:
                if isinstance(fct, str):
                    # skip special case string-functions (e.g. "mean")
                    wrapped_aggregation.append(fct)
                else:
                    # wrap function to prevent unexpected behavior of pd.DataFrame.agg
                    # otherwise, data is internally passed to apply element-wise instead of as whole series
                    # for user-defined functions: https://github.com/pandas-dev/pandas/issues/41768
                    wrapped_aggregation.append(_allow_only_series(fct))
            # agg function only accepts strings as identifiers for one-level columns
            if isinstance(key, tuple) and len(key) == 1:
                key = key[0]
            if not isinstance(key, (tuple, str)):
                raise ValueError(
                    f"The key {key} has an invalid type. It must either be a string or a tuple of strings."
                )
            agg_aggregations.setdefault(key, []).extend(wrapped_aggregation)
    return manual_aggregations, agg_aggregations


def _allow_only_series(func: callable) -> callable:
    # if data are passed to apply element-wise,
    # throw an error to ensure that they are processed as whole series
    @wraps(func)
    def wrapper(x: pd.Series) -> Any:
        if not isinstance(x, (pd.Series, pd.DataFrame)):
            raise TypeError("Only Series allowed as input.")
        return func(x)

    return wrapper


def _apply_manual_aggregations(df: pd.DataFrame, manual_aggregations: list[CustomOperation]) -> pd.Series:
    # apply manual aggregations
    manual_aggregation_results = []
    for agg in manual_aggregations:
        agg_functions = agg.function
        if not isinstance(agg_functions, list):
            agg_functions = [agg_functions]

        data = _get_data_from_identifier(df, agg.identifier, num_levels=None)
        for fct in agg_functions:
            result = fct(data)
            try:
                fct_name = fct.__name__
            except AttributeError as e:
                raise ValueError(
                    f"Transformation function {fct} applied for {agg.identifier} does not have a `__name__`-Attribute. "
                    "Please use a named function or assign a name."
                ) from e
            column_name = (agg.column_name,) if not isinstance(agg.column_name, tuple) else agg.column_name
            key = (fct_name, *column_name)
            manual_aggregation_results.append(pd.Series([result], index=pd.MultiIndex.from_tuples([key])))
    if manual_aggregation_results:
        try:
            _check_number_of_index_levels(manual_aggregation_results)
        except ValueError as e:
            raise ValueError(
                "Error in concatenating manual aggregation results. "
                "Please ensure that the `col_names` attribute has the same number of elements "
                "across all custom aggregations"
            ) from e
        manual_aggregation_results = pd.concat(manual_aggregation_results)
    return manual_aggregation_results


def _check_number_of_index_levels(agg_results: list[Union[pd.Series, pd.DataFrame]]) -> None:
    n_levels = [result.index.nlevels for result in agg_results]
    if len(set(n_levels)) > 1:
        raise ValueError(
            "Number of index levels in results is not consistent. "
            "Please ensure that all aggregation results have the same number of index levels."
        )


__all__ = [
    "categorize_intervals",
    "categorize_matches_with_min_overlap",
    "calculate_matched_gsd_performance_metrics",
    "calculate_unmatched_gsd_performance_metrics",
    "plot_categorized_intervals",
    "get_matching_intervals",
    "quantiles",
    "loa",
    "icc",
    "error",
    "rel_error",
    "abs_error",
    "abs_rel_error",
    "get_default_error_transformations",
    "get_default_aggregations",
    "apply_transformations",
    "apply_aggregations",
    "CustomOperation",
]
