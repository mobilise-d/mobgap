import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal

from gaitlink.gsd.evaluation import (
    _get_tn_intervals,
    calculate_gsd_performance_metrics,
    categorize_intervals,
    find_matches_with_min_overlap,
)


@pytest.fixture()
def intervals_example():
    return [[1, 3], [5, 7]]


@pytest.fixture()
def intervals_example_more_samples():
    return [[0, 5], [8, 13]]


@pytest.fixture()
def intervals_example_with_id(intervals_example):
    return pd.DataFrame([[1, 3, 0], [5, 7, 1]], columns=["start", "end", "id"]).set_index("id")


class TestCategorizeIntervals:
    """Tests for categorize_intervals method for gsd validation."""

    def test_raise_type_error_no_df(self, intervals_example):
        with pytest.raises(TypeError):
            categorize_intervals(intervals_example, intervals_example)

    def test_raise_value_error_wrong_input_columns(self, intervals_example):
        with pytest.raises(ValueError):
            # only default column names
            categorize_intervals(pd.DataFrame(intervals_example), pd.DataFrame(intervals_example))

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

    def test_partial_and_complete_overlap(self, intervals_example):
        interval_partial_and_complete_overlap_with_example = [[1, 3], [4, 6]]
        overlap = [[1, 3], [5, 6]]
        no_overlap_1 = [[6, 7]]
        no_overlap_2 = [[4, 5]]
        self._assert_equal_tp_fp_fn(
            intervals_example, interval_partial_and_complete_overlap_with_example, overlap, no_overlap_1, no_overlap_2
        )
        self._assert_equal_tp_fp_fn(
            interval_partial_and_complete_overlap_with_example, intervals_example, overlap, no_overlap_2, no_overlap_1
        )

    def test_partial_overlap_one(self, intervals_example):
        interval_partial_overlap_with_example = [[2, 4]]
        overlap = [[2, 3]]
        no_overlap_1 = [[1, 2], [5, 7]]
        no_overlap_2 = [[3, 4]]
        self._assert_equal_tp_fp_fn(
            intervals_example, interval_partial_overlap_with_example, overlap, no_overlap_1, no_overlap_2
        )
        self._assert_equal_tp_fp_fn(
            interval_partial_overlap_with_example, intervals_example, overlap, no_overlap_2, no_overlap_1
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
        result = categorize_intervals(
            pd.DataFrame(detected, columns=["start", "end"]), pd.DataFrame(reference, columns=["start", "end"])
        )
        expected_tp = pd.DataFrame(expected_tp, columns=["start", "end"])
        expected_fp = pd.DataFrame(expected_fp, columns=["start", "end"])
        expected_fn = pd.DataFrame(expected_fn, columns=["start", "end"])

        tp = result.query("match_type == 'tp'").drop(columns=["match_type"]).reset_index(drop=True)
        fp = result.query("match_type == 'fp'").drop(columns=["match_type"]).reset_index(drop=True)
        fn = result.query("match_type == 'fn'").drop(columns=["match_type"]).reset_index(drop=True)

        assert_frame_equal(tp, expected_tp, check_dtype=False)
        assert_frame_equal(fp, expected_fp, check_dtype=False)
        assert_frame_equal(fn, expected_fn, check_dtype=False)


class TestMatchIntervals:
    """`Tests for find_matches_with_min_overlap` method for gsd validation."""

    def test_raise_type_error_no_df(self, intervals_example):
        with pytest.raises(TypeError):
            find_matches_with_min_overlap(intervals_example, intervals_example)

    def test_raise_value_error_wrong_input_columns(self, intervals_example, intervals_example_with_id):
        with pytest.raises(ValueError):
            # only default columns
            find_matches_with_min_overlap(pd.DataFrame(intervals_example), pd.DataFrame(intervals_example_with_id))

    @pytest.mark.parametrize("min_overlap", [0, 0.5, 2])
    def test_raise_value_error_invalid_overlap(self, intervals_example_with_id, min_overlap):
        with pytest.raises(ValueError):
            find_matches_with_min_overlap(
                self._to_interval_df(intervals_example_with_id),
                self._to_interval_df(intervals_example_with_id),
                min_overlap,
            )

    def test_raise_value_error_invalid_index(self, intervals_example):
        index_not_unique = pd.DataFrame(intervals_example)
        index_not_unique["id"] = ["id"] * len(intervals_example)
        index_not_unique = index_not_unique.set_index("id")
        with pytest.raises(ValueError):
            find_matches_with_min_overlap(index_not_unique, index_not_unique, 1)

    def test_validation_all_tp(self, intervals_example):
        output = find_matches_with_min_overlap(
            self._to_interval_df(intervals_example), self._to_interval_df(intervals_example), 1
        )
        assert_array_equal(output.to_numpy(), intervals_example)

    def test_validation_all_tp_with_id(self, intervals_example_with_id):
        ref = intervals_example_with_id.copy()
        ref["id"] = ["ref"] * len(ref)
        output = find_matches_with_min_overlap(intervals_example_with_id, ref, 1)
        assert_frame_equal(output, intervals_example_with_id)

    def test_validation_all_false(self, intervals_example):
        intervals_no_overlap_with_example = [[0, 1], [3, 5]]
        output = find_matches_with_min_overlap(
            self._to_interval_df(intervals_example), self._to_interval_df(intervals_no_overlap_with_example), 1
        )
        assert output.empty
        output = find_matches_with_min_overlap(
            self._to_interval_df(intervals_no_overlap_with_example), self._to_interval_df(intervals_example), 1
        )
        assert output.empty

    def test_complete_overlap_one(self, intervals_example):
        intervals_partial_overlap_with_example = [[0, 3]]  # overlapping region is [[1, 3]]

        # threshold small enough for overlap
        output = find_matches_with_min_overlap(
            self._to_interval_df(intervals_example), self._to_interval_df(intervals_partial_overlap_with_example), 0.6
        )
        expected = [intervals_example[0]]
        assert_array_equal(output.to_numpy(), expected)

        output = find_matches_with_min_overlap(
            self._to_interval_df(intervals_partial_overlap_with_example), self._to_interval_df(intervals_example), 0.6
        )
        expected = intervals_partial_overlap_with_example
        assert_array_equal(output.to_numpy(), expected)

    def test_min_overlap_not_reached(self, intervals_example):
        intervals_partial_overlap_with_example = [[0, 3]]  # overlapping region is [[1, 3]]e
        output = find_matches_with_min_overlap(
            self._to_interval_df(intervals_example), self._to_interval_df(intervals_partial_overlap_with_example), 1
        )
        assert output.empty

        output = find_matches_with_min_overlap(
            self._to_interval_df(intervals_partial_overlap_with_example), self._to_interval_df(intervals_example), 1
        )
        assert output.empty

    def test_partial_overlap_start_and_end(self, intervals_example_more_samples):
        intervals_partial_overlap_with_example = [[2, 6], [7, 11]]  # overlapping regions are [[2, 5], [8, 11]]

        output = find_matches_with_min_overlap(
            self._to_interval_df(intervals_example_more_samples),
            self._to_interval_df(intervals_partial_overlap_with_example),
            0.6,
        )
        assert_array_equal(output.to_numpy(), intervals_example_more_samples)

        output = find_matches_with_min_overlap(
            self._to_interval_df(intervals_partial_overlap_with_example),
            self._to_interval_df(intervals_example_more_samples),
            0.6,
        )
        assert_array_equal(output.to_numpy(), intervals_partial_overlap_with_example)

    def test_partial_overlap_start_or_end(self, intervals_example_more_samples):
        intervals_partial_overlap_with_example = [[2, 5], [8, 11]]  # overlapping regions are [[2, 5], [8, 11]]

        output = find_matches_with_min_overlap(
            self._to_interval_df(intervals_example_more_samples),
            self._to_interval_df(intervals_partial_overlap_with_example),
            0.6,
        )
        assert_array_equal(output.to_numpy(), intervals_example_more_samples)

        output = find_matches_with_min_overlap(
            self._to_interval_df(intervals_partial_overlap_with_example),
            self._to_interval_df(intervals_example_more_samples),
            0.6,
        )
        assert_array_equal(output.to_numpy(), intervals_partial_overlap_with_example)

    def test_several_intervals_overlap_one(self, intervals_example_more_samples):
        interval_several_overlaps_with_example = [[0, 9]]  # overlapping regions are [[0, 5], [8, 9]]
        output = find_matches_with_min_overlap(
            self._to_interval_df(intervals_example_more_samples),
            self._to_interval_df(interval_several_overlaps_with_example),
            5 / 9,
        )
        assert_array_equal(output.to_numpy(), [intervals_example_more_samples[0]])

        output = find_matches_with_min_overlap(
            self._to_interval_df(interval_several_overlaps_with_example),
            self._to_interval_df(intervals_example_more_samples),
            5 / 9,
        )
        assert_array_equal(output.to_numpy(), interval_several_overlaps_with_example)

    def test_several_intervals_overlap_several(self, intervals_example_more_samples):
        intervals_several_overlaps_with_example = [[0, 9], [9, 12]]  # overlapping regions are [[0, 5], [8, 9], [9, 12]]
        output = find_matches_with_min_overlap(
            self._to_interval_df(intervals_example_more_samples),
            self._to_interval_df(intervals_several_overlaps_with_example),
            5 / 9,
        )
        assert_array_equal(output.to_numpy(), intervals_example_more_samples)

        output = find_matches_with_min_overlap(
            self._to_interval_df(intervals_several_overlaps_with_example),
            self._to_interval_df(intervals_example_more_samples),
            5 / 9,
        )
        assert_array_equal(output.to_numpy(), intervals_several_overlaps_with_example)

    @staticmethod
    def _to_interval_df(array):
        return pd.DataFrame(array, columns=["start", "end"])


class TestGetTnIntervals:
    """Tests for get_tn_intervals method for gsd validation."""

    def test_no_tn(self):
        categorized_intervals = pd.DataFrame(
            [[0, 1, "tp"], [1, 2, "fp"], [2, 4, "fn"]], columns=["start", "end", "match_type"]
        )
        tn_intervals = _get_tn_intervals(categorized_intervals, n_overall_samples=5)
        assert tn_intervals.empty

    def test_start_tn(self):
        categorized_intervals = pd.DataFrame([[1, 2, "fp"], [2, 4, "fn"]], columns=["start", "end", "match_type"])
        tn_expected = pd.DataFrame([[0, 1, "tn"]], columns=["start", "end", "match_type"])
        tn_intervals = _get_tn_intervals(categorized_intervals, n_overall_samples=5)
        assert_frame_equal(tn_intervals, tn_expected)

    def test_end_tn(self):
        categorized_intervals = pd.DataFrame([[0, 1, "tp"], [1, 2, "fp"]], columns=["start", "end", "match_type"])
        tn_expected = pd.DataFrame([[2, 4, "tn"]], columns=["start", "end", "match_type"])
        tn_intervals = _get_tn_intervals(categorized_intervals, n_overall_samples=5)
        assert_frame_equal(tn_intervals, tn_expected)

    def test_several_tn(self):
        categorized_intervals = pd.DataFrame(
            [[1, 2, "fp"], [3, 4, "fn"], [7, 8, "fp"], [10, 12, "fn"]], columns=["start", "end", "match_type"]
        )
        tn_expected = pd.DataFrame(
            [[0, 1, "tn"], [2, 3, "tn"], [4, 7, "tn"], [8, 10, "tn"], [12, 14, "tn"]],
            columns=["start", "end", "match_type"],
        )
        tn_intervals = _get_tn_intervals(categorized_intervals, n_overall_samples=15)
        assert_frame_equal(tn_intervals, tn_expected)


class TestGsdPerformanceMetrics:
    """Tests for calculate_gsd_performance_metrics method for gsd validation."""

    def test_output(self, snapshot):
        reference = pd.DataFrame([[0, 10], [15, 25]], columns=["start", "end"])
        detected = pd.DataFrame([[0, 10], [20, 30]], columns=["start", "end"])
        metrics = calculate_gsd_performance_metrics(detected, reference, 10, 100)
        snapshot.assert_match(pd.DataFrame(metrics, index=[0]), "metrics")

    def test_output_no_matches(self, snapshot):
        reference = pd.DataFrame([[0, 10], [15, 25]], columns=["start", "end"])
        detected = pd.DataFrame([[10, 15], [30, 35]], columns=["start", "end"])
        metrics = calculate_gsd_performance_metrics(detected, reference, 10, 100)
        snapshot.assert_match(pd.DataFrame(metrics, index=[0]), "metrics_no_match")

    def test_raise_wrong_num_samples(self, intervals_example_with_id):
        with pytest.raises(ValueError):
            calculate_gsd_performance_metrics(
                intervals_example_with_id, intervals_example_with_id, 10, n_overall_samples=2
            )
