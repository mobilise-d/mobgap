"""Class to validate gait sequence detection results."""
from typing import NamedTuple

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
    Validate detected gait sequence intervals against a reference.

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
    """
    # check if input is a dataframe with two columns
    if not isinstance(gsd_list_detected, pd.DataFrame) or not isinstance(gsd_list_reference, pd.DataFrame):
        raise TypeError("`gsd_list_detected` and `gsd_list_reference` must be of type `pandas.DataFrame`.")
    if gsd_list_detected.shape[1] >= 2 or gsd_list_reference.shape[1] >= 2:
        raise ValueError(
            "`gsd_list_detected` and `gsd_list_reference` must have at least two columns, with the first column "
            "containing the start and the second column containing the end indices of the gait sequences."
        )

    # Create Interval Trees
    reference_tree = IntervalTree.from_tuples(gsd_list_reference.values)
    detected_tree = IntervalTree.from_tuples(gsd_list_detected.values)

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

    # Convert lists to DataFrames
    tp_intervals = pd.DataFrame(merge_intervals(tp_intervals), columns=["start", "end"])
    fp_intervals = pd.DataFrame(merge_intervals(fp_intervals), columns=["start", "end"])
    fn_intervals = pd.DataFrame(merge_intervals(fn_intervals), columns=["start", "end"])

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

    """
    def _find_matches_with_min_overlap(self) -> Sequence[Tuple[int, int]]:
      '''
      Find all matches of interval_seq_query in interval_seq with at least overlap_threshold overlap.

        Note, that the threshold is enforced in both directions!
        The overlap region must cover at least the threshold value of both matched intervals.

        Note, we assume that `interval_seq` has no overlaps, but we don't enforce it!

      interval_overlap_threshold: float
            The minimum relative overlap between a gait sequence in the input and reference to be considered a match.
            Must be larger than 0 and smaller than or equal to 1.
        match_overlap_threshold: float
            The minimum percentage of a detected gait sequence that needs to overlap with a reference sequence to be
            considered a match.
            Must be larger than 0 and smaller than or equal to 1.
        '''
        if self.interval_overlap_threshold <= 0.5:
            raise ValueError(
                "overlap_threshold must be greater than 0.5."
                "Otherwise multiple matches between intervals "
                "are possible."
            )
        if self.interval_overlap_threshold > 1:
            raise ValueError("overlap_threshold must be less than 1." "Otherwise no matches can be returned.")
        tree = IntervalTree.from_tuples(self.gsd_list_reference.values)

        final_matches = []
        for interval in self.gsd_list.values:
            matches = tree[interval[0]: interval[1]]
            if len(matches) > 0:
                for match in matches:
                    # First calculate the absolute overlap
                    absolute_overlap = match.overlap_size(interval[0], interval[1])
                    # Then calculate the relative overlap
                    relative_overlap_interval = absolute_overlap / (interval[1] - interval[0])
                    relative_overlap_match = absolute_overlap / (match[1] - match[0])
                    if relative_overlap_interval >= self.interval_overlap_threshold and relative_overlap_match
                    >= self.match_overlap_threshold:
                        final_matches.append((match[0], match[1]))
                        break
                else:
                    final_matches.append(None)
            else:
                final_matches.append(None)

        return final_matches
        """
