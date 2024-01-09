import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from gaitlink.gsd.validation import categorize_intervals


@pytest.fixture()
def intervals_example():
    return [[1, 3], [5, 7]]


@pytest.fixture()
def intervals_example_as_df(intervals_example):
    return pd.DataFrame([[1, 3], [5, 7]], columns=["start", "end"])


class TestGsdValidation:
    """Tests for Gsd validation."""

    def test_validation_all_tp(self, intervals_example):
        expected_tp, expected_fp, expected_fn = intervals_example, [], []
        self._assert_equal_tp_fp_fn(intervals_example, intervals_example, expected_tp, expected_fp, expected_fn)

    def test_validation_all_false(self, intervals_example):
        intervals_no_overlap_with_example = [[0, 1], [3, 5]]
        overlap = []
        self._assert_equal_tp_fp_fn(
            intervals_example,
            intervals_no_overlap_with_example,
            overlap,
            intervals_example,
            intervals_no_overlap_with_example,
        )
        self._assert_equal_tp_fp_fn(
            intervals_no_overlap_with_example,
            intervals_example,
            overlap,
            intervals_no_overlap_with_example,
            intervals_example,
        )

    def test_partial_overlap_start_and_end(self, intervals_example):
        intervals_partial_overlap_with_example = [[2, 4], [6, 8]]
        overlap = [[2, 3], [6, 7]]
        no_overlap_1 = [[1, 2], [5, 6]]
        no_overlap_2 = [[3, 4], [7, 8]]
        self._assert_equal_tp_fp_fn(
            intervals_example, intervals_partial_overlap_with_example, overlap, no_overlap_1, no_overlap_2
        )
        self._assert_equal_tp_fp_fn(
            intervals_partial_overlap_with_example, intervals_example, overlap, no_overlap_2, no_overlap_1
        )

    def test_partial_overlap_start_or_end(self, intervals_example):
        intervals_partial_overlap_with_example = [[2, 3], [5, 8]]
        overlap = [[2, 3], [5, 7]]
        fp_overlap = [[1, 2]]
        fn_overlap = [[7, 8]]
        self._assert_equal_tp_fp_fn(
            intervals_example, intervals_partial_overlap_with_example, overlap, fp_overlap, fn_overlap
        )
        self._assert_equal_tp_fp_fn(
            intervals_partial_overlap_with_example, intervals_example, overlap, fn_overlap, fp_overlap
        )

    def test_several_intervals_overlap_one(self, intervals_example):
        interval_several_overlaps_with_example = [[0, 9]]
        fp_overlap = []
        fn_overlap = [[0, 1], [3, 5], [7, 9]]
        self._assert_equal_tp_fp_fn(
            intervals_example, interval_several_overlaps_with_example, intervals_example, fp_overlap, fn_overlap
        )
        self._assert_equal_tp_fp_fn(
            interval_several_overlaps_with_example, intervals_example, intervals_example, fn_overlap, fp_overlap
        )

    def test_several_intervals_partially_overlap_one(self, intervals_example):
        interval_partial_overlaps_with_example = [[2, 6]]
        overlap = [[2, 3], [5, 6]]
        fp_overlap = [[1, 2], [6, 7]]
        fn_overlap = [[3, 5]]
        self._assert_equal_tp_fp_fn(
            intervals_example, interval_partial_overlaps_with_example, overlap, fp_overlap, fn_overlap
        )
        self._assert_equal_tp_fp_fn(
            interval_partial_overlaps_with_example, intervals_example, overlap, fn_overlap, fp_overlap
        )

    def test_several_intervals_partially_overlap_several(self):
        intervals_example = [[0, 3], [4, 7], [8, 9]]
        intervals_example_shifted = [[0, 1], [2, 5], [6, 9]]
        overlap = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        fp_overlap = [[1, 2], [5, 6]]
        fn_overlap = [[3, 4], [7, 8]]
        self._assert_equal_tp_fp_fn(intervals_example, intervals_example_shifted, overlap, fp_overlap, fn_overlap)
        self._assert_equal_tp_fp_fn(intervals_example_shifted, intervals_example, overlap, fn_overlap, fp_overlap)

    @staticmethod
    def _assert_equal_tp_fp_fn(detected, reference, expected_tp, expected_fp, expected_fn):
        result = categorize_intervals(pd.DataFrame(detected), pd.DataFrame(reference))
        expected_tp = pd.DataFrame(expected_tp, columns=["start", "end"])
        expected_fp = pd.DataFrame(expected_fp, columns=["start", "end"])
        expected_fn = pd.DataFrame(expected_fn, columns=["start", "end"])

        tp = result.tp_intervals
        fp = result.fp_intervals
        fn = result.fn_intervals

        assert_frame_equal(tp, expected_tp)
        assert_frame_equal(fp, expected_fp)
        assert_frame_equal(fn, expected_fn)
