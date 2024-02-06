"""Class to validate gait sequence detection results."""
from typing import NamedTuple

import numpy as np
import pandas as pd
from gaitmap.utils.array_handling import merge_intervals
from intervaltree import IntervalTree
from intervaltree.interval import Interval


class CategorizedIntervals(NamedTuple):
    """Helper class to store the results of the sample-wise validation."""

    tp_intervals: pd.DataFrame
    fp_intervals: pd.DataFrame
    fn_intervals: pd.DataFrame


def categorize_intervals(gsd_list_detected: pd.DataFrame, gsd_list_reference: pd.DataFrame) -> CategorizedIntervals:
    """
    Validate detected gait sequence intervals against a reference on a sample-wise level.

    The detected and reference dataframes are expected to have columns namend "start" and "end" containing the
    start and end indices of the respective gait sequences.
    Each sample from the detected interval list is categorized as true positive (TP),
    false positive (FP) or false negative (FN).
    The results are concatenated into three result dataframes `tp_intervals`, `fp_intervals` and `fn_intervals`,
    which are returned as a NamedTuple.

    Parameters
    ----------
    gsd_list_detected: pd.DataFrame
       Each row contains a detected gait sequence interval as output from the GSD algorithms.
       The respective start index is stored in the first and the stop index in the second column.
    gsd_list_reference: pd.DataFrame
       Gold standard to validate the detected gait sequences against.
       Should have the same format as `gsd_list_detected`.

    Returns
    -------
    CategorizedIntervals
        A NamedTuple containing the three result dataframes `tp_intervals`,
        `fp_intervals` and `fn_intervals` as attributes.

    Examples
    --------
    >>> from gaitlink.gsd.validation import categorize_intervals
    >>> detected = pd.DataFrame([[0, 10], [20, 30]], columns=["start", "end"])
    >>> reference = pd.DataFrame([[0, 10], [15, 25]], columns=["start", "end"])
    >>> result = categorize_intervals(detected, reference)
    >>> result.tp_intervals
       start  end
    0      0   10
    1     20   25
    """
    # check if input is a dataframe with two columns
    if not isinstance(gsd_list_detected, pd.DataFrame) or not isinstance(gsd_list_reference, pd.DataFrame):
        raise TypeError("`gsd_list_detected` and `gsd_list_reference` must be of type `pandas.DataFrame`.")
    # check if start and end columns are present
    if not all(key in gsd_list_detected.columns for key in ["start", "end"]) and not all(
        key in gsd_list_reference.columns for key in ["start", "end"]
    ):
        raise ValueError("`gsd_list_detected` must have columns named 'start' and 'end'.")

    # Create Interval Trees
    reference_tree = IntervalTree.from_tuples(gsd_list_reference[["start", "end"]].to_numpy())
    detected_tree = IntervalTree.from_tuples(gsd_list_detected[["start", "end"]].to_numpy())

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

    # convert results to pandas DataFrame
    tp_intervals = pd.DataFrame(
        merge_intervals(np.array(tp_intervals)) if len(tp_intervals) != 0 else tp_intervals, columns=["start", "end"]
    )
    fp_intervals = pd.DataFrame(
        merge_intervals(np.array(fp_intervals)) if len(fp_intervals) != 0 else fp_intervals, columns=["start", "end"]
    )
    fn_intervals = pd.DataFrame(
        merge_intervals(np.array(fn_intervals)) if len(fn_intervals) != 0 else fn_intervals, columns=["start", "end"]
    )

    result = CategorizedIntervals(tp_intervals=tp_intervals, fp_intervals=fp_intervals, fn_intervals=fn_intervals)

    return result


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
    # check if input is a dataframe with two columns
    if not isinstance(gsd_list_detected, pd.DataFrame) or not isinstance(gsd_list_reference, pd.DataFrame):
        raise TypeError("`gsd_list_detected` and `gsd_list_reference` must be of type `pandas.DataFrame`.")

    # check if start and end columns are present
    if not all(key in gsd_list_detected.columns for key in ["start", "end"]) or not all(
        key in gsd_list_reference.columns for key in ["start", "end"]
    ):
        raise ValueError("`gsd_list_detected` must have columns named 'start' and 'end'.")

    # check if index is unique
    if not gsd_list_detected.index.is_unique:
        raise ValueError("`gsd_list_detected` must have a unique index to .")

    if overlap_threshold <= 0.5:
        raise ValueError(
            "overlap_threshold must be greater than 0.5."
            "Otherwise multiple matches between intervals "
            "are possible."
        )
    if overlap_threshold > 1:
        raise ValueError("overlap_threshold must be less than 1." "Otherwise no matches can be returned.")

    tree = IntervalTree.from_tuples(gsd_list_detected.reset_index(names="id")[["start", "end", "id"]].to_numpy())
    final_matches = []

    for _, interval in gsd_list_reference[["start", "end"]].iterrows():
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
