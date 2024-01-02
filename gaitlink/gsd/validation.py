import pandas as pd
from intervaltree import IntervalTree
from intervaltree.interval import Interval


class GsdValidator:
    """
    Parameters
    ----------
    gap_size : int
        Maximum number of samples between two subsequent intervals to be considered as one interval.

    Attributes
    ----------
    tp_intervals_: pd.DataFrame
        True positive intervals.
    fp_intervals_: pd.DataFrame
        False positive intervals.
    fn_intervals_: pd.DataFrame
        False negative intervals.
    """

    gsd_list_detected: pd.DataFrame
    gsd_list_reference: pd.DataFrame
    tp_intervals_: pd.DataFrame
    fp_intervals_: pd.DataFrame
    fn_intervals_: pd.DataFrame
    tp_samples_: int
    fp_samples_: int
    fn_samples_: int

    def __init__(self, gap_size=0):
        self.gap_size = gap_size  # TODO should we use this?

    def validate(self, gsd_list_detected: pd.DataFrame, gsd_list_reference: pd.DataFrame):
        """
        Parameters
        ----------
        gsd_list_detected: pd.DataFrame
            Each row contains a detected gait sequence interval as output from the GSD algorithms.
            The respective start index is stored in the first and the stop index in the second column.
        gsd_list_reference: pd.DataFrame
            Gold standard to validate the detected gait sequences against.
            Should have the same format as `gsd_list_detected`.
        """
        self.gsd_list_detected = gsd_list_detected
        self.gsd_list_reference = gsd_list_reference
        self.tp_intervals_, self.fp_intervals_, self.fn_intervals_ = self._categorize_intervals()
        self.tp_samples_, self.fp_samples_, self.fn_samples_ = (
            self._count_samples_in_intervals(self.tp_intervals_),
            self._count_samples_in_intervals(self.fp_intervals_),
            self._count_samples_in_intervals(self.fn_intervals_),
        )
        # self.matches_ = self._find_matches_with_min_overlap()
        return self

    def _categorize_intervals(self):
        # Create Interval Trees
        reference_tree = IntervalTree.from_tuples(self.gsd_list_reference.values)
        detected_tree = IntervalTree.from_tuples(self.gsd_list_detected.values)

        # Prepare DataFrames for TP, FP, FN
        tp_intervals = []
        fp_intervals = []
        fn_intervals = []

        # Calculate TP and FP
        for interval in detected_tree:
            overlaps = sorted(reference_tree.overlap(interval.begin, interval.end))
            if overlaps:
                fp_matches = self._get_false_matches_from_overlap_data(overlaps, interval)
                fp_intervals.extend(fp_matches)

                for i, overlap in enumerate(overlaps):
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
                fn_matches = self._get_false_matches_from_overlap_data(overlaps, interval)
                fn_intervals.extend(fn_matches)

        # Convert lists to DataFrames
        tp_intervals = pd.DataFrame(self._sort_intervals(tp_intervals), columns=["start", "end"])
        fp_intervals = pd.DataFrame(self._sort_intervals(fp_intervals), columns=["start", "end"])
        fn_intervals = pd.DataFrame(self._sort_intervals(fn_intervals), columns=["start", "end"])

        return tp_intervals, fp_intervals, fn_intervals

    @staticmethod
    def _sort_intervals(intervals: list[list[int]]) -> Sequence[list[int]]:
        if len(intervals) == 0:
            return intervals
        return np.sort(intervals, axis=0, kind="stable")

    @staticmethod
    def _count_samples_in_intervals(intervals: pd.DataFrame) -> int:
        return (intervals["end"] - intervals["start"]).sum()

    @staticmethod
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
                    if relative_overlap_interval >= self.interval_overlap_threshold and relative_overlap_match >= self.match_overlap_threshold:
                        final_matches.append((match[0], match[1]))
                        break
                else:
                    final_matches.append(None)
            else:
                final_matches.append(None)

        return final_matches
        """

    def print_results(self):
        print("True Positives:\n\n", self.tp_intervals_)
        print("\nFalse Positives:\n\n", self.fp_intervals_)
        print("\nFalse Negatives:\n\n", self.fn_intervals_)
        print("-" * 30)


if __name__ == "__main__":
    detected = pd.DataFrame([[2, 4], [5, 7]], columns=["start", "end"])
    reference = pd.DataFrame([[1, 8]], columns=["start", "end"])
    validator = GsdValidator()
    validator.validate(detected, reference)
    validator.print_results()
    validator.validate(reference, detected)
    validator.print_results()
