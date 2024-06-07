from itertools import product
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal, assert_series_equal

from mobgap.gsd.evaluation import (
    CustomOperation,
    _get_tn_intervals,
    abs_error,
    abs_rel_error,
    apply_aggregations,
    apply_transformations,
    calculate_matched_gsd_performance_metrics,
    calculate_unmatched_gsd_performance_metrics,
    categorize_intervals,
    categorize_matches_with_min_overlap,
    combine_det_with_ref_without_matching,
    error,
    get_default_aggregations,
    get_default_error_transformations,
    get_matching_gs,
    icc,
    loa,
    quantiles,
    rel_error,
)


@pytest.fixture()
def intervals_example():
    return [[1, 3], [5, 7]]


@pytest.fixture()
def intervals_example_more_samples():
    return [[0, 5], [8, 13]]


@pytest.fixture()
def intervals_example_with_id():
    return pd.DataFrame([[1, 3, 0], [5, 7, 1]], columns=["start", "end", "id"]).set_index("id")


@pytest.fixture()
def dmo_df():
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["metric_a", "metric_b", "metric_c"])
    df.index = pd.MultiIndex.from_tuples([("a", 1), ("a", 2)], names=["group", "wb_id"])
    return df


@pytest.fixture()
def combined_det_ref_dmo_df():
    metrics = [
        "cadence_spm",
        "duration_s",
        "n_steps",
        "n_turns",
        "stride_duration_s",
        "stride_length_m",
        "walking_speed_mps",
        "wb_id",
    ]
    data = np.zeros((2, len(metrics) * 2))
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_product([metrics, ["detected", "reference"]])
    return df


@pytest.fixture()
def combined_det_ref_dmo_df_with_errors():
    metrics = [
        "cadence_spm",
        "duration_s",
        "n_steps",
        "n_turns",
        "stride_duration_s",
        "stride_length_m",
        "walking_speed_mps",
        "wb_id",
    ]
    data = np.zeros((3, len(metrics) * 6))
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_product(
        [metrics, ["detected", "reference", "error", "abs_error", "rel_error", "abs_rel_error"]]
    )
    return df


@pytest.fixture()
def matches_df():
    return pd.DataFrame(
        {
            "gs_id_detected": [("a", 1), ("a", 2), np.nan],
            "gs_id_reference": [("a", 2), np.nan, ("a", 1)],
            "match_type": ["tp", "fp", "fn"],
        }
    )


class TestCategorizeIntervals:
    """Tests for categorize_intervals method for gsd validation."""

    def test_raise_type_error_no_df(self, intervals_example):
        with pytest.raises(TypeError):
            categorize_intervals(gsd_list_detected=intervals_example, gsd_list_reference=intervals_example)

    def test_raise_value_error_wrong_input_columns(self, intervals_example):
        with pytest.raises(ValueError):
            # only default column names
            categorize_intervals(
                gsd_list_detected=pd.DataFrame(intervals_example), gsd_list_reference=pd.DataFrame(intervals_example)
            )

    def test_raise_wrong_num_samples(self, intervals_example_with_id):
        with pytest.raises(ValueError):
            categorize_intervals(
                gsd_list_detected=pd.DataFrame(intervals_example_with_id),
                gsd_list_reference=pd.DataFrame(intervals_example_with_id),
                n_overall_samples=2,
            )

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

    def test_several_intervals_partially_overlap_several_with_tn(self):
        intervals_example = [[0, 3], [4, 7], [8, 9]]
        intervals_example_shifted = [[0, 1], [2, 5], [6, 9]]
        overlap = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        fp_overlap = [[1, 2], [5, 6]]
        fn_overlap = [[3, 4], [7, 8]]
        tn_overlap = [[9, 11]]
        self._assert_equal_tp_fp_fn(
            intervals_example,
            intervals_example_shifted,
            overlap,
            fp_overlap,
            fn_overlap,
            tn_overlap,
            n_overall_samples=12,
        )
        self._assert_equal_tp_fp_fn(
            intervals_example_shifted,
            intervals_example,
            overlap,
            fn_overlap,
            fp_overlap,
            tn_overlap,
            n_overall_samples=12,
        )

    @staticmethod
    def _assert_equal_tp_fp_fn(
        detected, reference, expected_tp, expected_fp, expected_fn, expected_tn=None, n_overall_samples=None
    ):
        result = categorize_intervals(
            gsd_list_detected=pd.DataFrame(detected, columns=["start", "end"]),
            gsd_list_reference=pd.DataFrame(reference, columns=["start", "end"]),
            n_overall_samples=n_overall_samples,
        )
        expected_tp = pd.DataFrame(expected_tp, columns=["start", "end"])
        expected_fp = pd.DataFrame(expected_fp, columns=["start", "end"])
        expected_fn = pd.DataFrame(expected_fn, columns=["start", "end"])
        expected_tn = (
            pd.DataFrame(expected_tn, columns=["start", "end"])
            if expected_tn is not None
            else pd.DataFrame(columns=["start", "end"])
        )

        tp = result.query("match_type == 'tp'").drop(columns=["match_type"]).reset_index(drop=True)
        fp = result.query("match_type == 'fp'").drop(columns=["match_type"]).reset_index(drop=True)
        fn = result.query("match_type == 'fn'").drop(columns=["match_type"]).reset_index(drop=True)
        tn = result.query("match_type == 'tn'").drop(columns=["match_type"]).reset_index(drop=True)

        assert_frame_equal(tp, expected_tp, check_dtype=False)
        assert_frame_equal(fp, expected_fp, check_dtype=False)
        assert_frame_equal(fn, expected_fn, check_dtype=False)
        assert_frame_equal(tn, expected_tn, check_dtype=False)


class TestMatchIntervals:
    """`Tests for categorize_matches_with_min_overlap` method for gsd validation."""

    def test_raise_type_error_no_df(self, intervals_example):
        with pytest.raises(TypeError):
            categorize_matches_with_min_overlap(
                gsd_list_detected=intervals_example, gsd_list_reference=intervals_example
            )

    def test_raise_value_error_wrong_input_columns(self, intervals_example, intervals_example_with_id):
        with pytest.raises(ValueError):
            # only default columns
            categorize_matches_with_min_overlap(
                gsd_list_detected=pd.DataFrame(intervals_example),
                gsd_list_reference=pd.DataFrame(intervals_example_with_id),
            )

    @pytest.mark.parametrize("min_overlap", [0, 0.5, 2])
    def test_raise_value_error_invalid_overlap(self, intervals_example_with_id, min_overlap):
        with pytest.raises(ValueError):
            categorize_matches_with_min_overlap(
                gsd_list_detected=self._to_interval_df(intervals_example_with_id),
                gsd_list_reference=self._to_interval_df(intervals_example_with_id),
                overlap_threshold=min_overlap,
            )

    def test_raise_value_error_invalid_index(self, intervals_example_with_id):
        index_not_unique = pd.DataFrame(intervals_example_with_id)
        index_not_unique["id"] = ["id"] * len(intervals_example_with_id)
        index_not_unique = index_not_unique.set_index("id")
        with pytest.raises(ValueError):
            categorize_matches_with_min_overlap(
                gsd_list_detected=index_not_unique, gsd_list_reference=index_not_unique, overlap_threshold=1
            )

    def test_input_multiindex_warning(self, intervals_example_with_id):
        multiindex = intervals_example_with_id.copy()
        multiindex.index = pd.MultiIndex.from_tuples([("a", 1), ("a", 2)], names=["something", "gsd_id"])
        with pytest.warns(Warning):
            categorize_matches_with_min_overlap(
                gsd_list_detected=multiindex, gsd_list_reference=intervals_example_with_id
            )
        with pytest.warns(Warning):
            categorize_matches_with_min_overlap(
                gsd_list_detected=intervals_example_with_id, gsd_list_reference=multiindex
            )
        with pytest.warns(Warning):
            categorize_matches_with_min_overlap(gsd_list_detected=multiindex, gsd_list_reference=multiindex)

    def test_input_multiindex_warning_suppressed(self, intervals_example_with_id):
        multiindex = intervals_example_with_id.copy()
        multiindex.index = pd.MultiIndex.from_tuples([("a", 1), ("a", 2)], names=["something", "ic_id"])
        with pytest.warns(None) as record:
            categorize_matches_with_min_overlap(
                gsd_list_detected=multiindex, gsd_list_reference=intervals_example_with_id, multiindex_warning=False
            )
            assert len(record) == 0
        with pytest.warns(None) as record:
            categorize_matches_with_min_overlap(
                gsd_list_detected=intervals_example_with_id, gsd_list_reference=multiindex, multiindex_warning=False
            )
            assert len(record) == 0
        with pytest.warns(None) as record:
            categorize_matches_with_min_overlap(
                gsd_list_detected=multiindex, gsd_list_reference=multiindex, multiindex_warning=False
            )
            assert len(record) == 0

    def test_validation_all_tp(self, intervals_example):
        matches = categorize_matches_with_min_overlap(
            gsd_list_detected=self._to_interval_df(intervals_example),
            gsd_list_reference=self._to_interval_df(intervals_example),
            overlap_threshold=1,
        )
        assert len(matches) == len(intervals_example)
        assert_array_equal(matches.query("match_type == 'tp'")["gs_id_detected"].to_numpy(), [0, 1])
        assert_array_equal(matches.query("match_type == 'tp'")["gs_id_reference"].to_numpy(), [0, 1])

    def test_input_multiindex(self, intervals_example_with_id):
        ic_list_multiindex = intervals_example_with_id.copy()
        ic_list_multiindex.index = pd.MultiIndex.from_tuples(
            [("a", 1), ("a", 2)], names=["something", "something_else"]
        )
        matches = categorize_matches_with_min_overlap(
            gsd_list_detected=ic_list_multiindex, gsd_list_reference=ic_list_multiindex
        )
        assert np.all(matches["match_type"] == "tp")
        assert len(matches) == len(ic_list_multiindex)
        assert_array_equal(matches["gs_id_detected"].to_numpy(), ic_list_multiindex.index.to_flat_index())
        assert_array_equal(matches["gs_id_reference"].to_numpy(), ic_list_multiindex.index.to_flat_index())

    def test_validation_all_tp_with_id(self, intervals_example_with_id):
        ref = intervals_example_with_id.copy()
        ref["id"] = ["ref"] * len(ref)
        matches = categorize_matches_with_min_overlap(
            gsd_list_detected=intervals_example_with_id, gsd_list_reference=ref, overlap_threshold=1
        )
        assert len(matches) == len(intervals_example_with_id)
        assert_array_equal(
            matches.query("match_type == 'tp'")["gs_id_detected"].to_numpy(), intervals_example_with_id.index
        )
        assert_array_equal(
            matches.query("match_type == 'tp'")["gs_id_reference"].to_numpy(), intervals_example_with_id.index
        )

    def test_validation_all_false(self, intervals_example):
        intervals_no_overlap_with_example = [[0, 1], [3, 5]]
        matches = categorize_matches_with_min_overlap(
            gsd_list_detected=self._to_interval_df(intervals_example),
            gsd_list_reference=self._to_interval_df(intervals_no_overlap_with_example),
            overlap_threshold=1,
        )
        assert len(matches) == len(intervals_example) + len(intervals_no_overlap_with_example)
        assert_array_equal(
            matches.query("match_type == 'fp'")["gs_id_reference"].to_numpy().astype(float), [np.nan, np.nan]
        )
        assert_array_equal(matches.query("match_type == 'fp'")["gs_id_detected"].to_numpy(), [0, 1])
        assert_array_equal(matches.query("match_type == 'fn'")["gs_id_reference"].to_numpy(), [0, 1])
        assert_array_equal(
            matches.query("match_type == 'fn'")["gs_id_detected"].to_numpy().astype(float), [np.nan, np.nan]
        )

    def test_complete_overlap_one(self, intervals_example):
        intervals_partial_overlap_with_example = [[0, 3]]  # overlapping region is [[1, 3]]

        # threshold small enough for overlap
        matches = categorize_matches_with_min_overlap(
            gsd_list_detected=self._to_interval_df(intervals_example),
            gsd_list_reference=self._to_interval_df(intervals_partial_overlap_with_example),
            overlap_threshold=0.6,
        )

        assert len(matches) == len(intervals_example)
        assert_array_equal(matches.query("match_type == 'tp'")["gs_id_reference"].to_numpy(), [0])
        assert_array_equal(matches.query("match_type == 'tp'")["gs_id_detected"].to_numpy(), [0])
        assert_array_equal(matches.query("match_type == 'fp'")["gs_id_reference"].to_numpy().astype(float), [np.nan])
        assert_array_equal(matches.query("match_type == 'fp'")["gs_id_detected"].to_numpy(), [1])

    def test_min_overlap_not_reached(self, intervals_example):
        intervals_partial_overlap_with_example = [[0, 3]]  # overlapping region is [[1, 3]]
        matches = categorize_matches_with_min_overlap(
            gsd_list_detected=self._to_interval_df(intervals_example),
            gsd_list_reference=self._to_interval_df(intervals_partial_overlap_with_example),
            overlap_threshold=1,
        )

        assert len(matches) == len(intervals_example) + len(intervals_partial_overlap_with_example)
        assert_array_equal(
            matches.query("match_type == 'fp'")["gs_id_reference"].to_numpy().astype(float), [np.nan, np.nan]
        )
        assert_array_equal(matches.query("match_type == 'fp'")["gs_id_detected"].to_numpy(), [0, 1])
        assert_array_equal(matches.query("match_type == 'fn'")["gs_id_reference"].to_numpy(), [0])

    def test_partial_overlap_start_and_end(self, intervals_example_more_samples):
        intervals_partial_overlap_with_example = [[2, 6], [7, 11]]  # overlapping regions are [[2, 5], [8, 11]]

        matches = categorize_matches_with_min_overlap(
            gsd_list_detected=self._to_interval_df(intervals_example_more_samples),
            gsd_list_reference=self._to_interval_df(intervals_partial_overlap_with_example),
            overlap_threshold=0.6,
        )

        assert len(matches) == len(intervals_example_more_samples)
        assert_array_equal(matches.query("match_type == 'tp'")["gs_id_reference"].to_numpy(), [0, 1])
        assert_array_equal(matches.query("match_type == 'tp'")["gs_id_detected"].to_numpy(), [0, 1])

    def test_partial_overlap_start_or_end(self, intervals_example_more_samples):
        intervals_partial_overlap_with_example = [[2, 5], [8, 11]]  # overlapping regions are [[2, 5], [8, 11]]

        matches = categorize_matches_with_min_overlap(
            gsd_list_detected=self._to_interval_df(intervals_example_more_samples),
            gsd_list_reference=self._to_interval_df(intervals_partial_overlap_with_example),
            overlap_threshold=0.6,
        )

        assert len(matches) == len(intervals_example_more_samples)
        assert_array_equal(matches.query("match_type == 'tp'")["gs_id_reference"].to_numpy(), [0, 1])
        assert_array_equal(matches.query("match_type == 'tp'")["gs_id_detected"].to_numpy(), [0, 1])

    def test_several_intervals_overlap_one(self, intervals_example_more_samples):
        interval_several_overlaps_with_example = [[0, 9]]  # overlapping regions are [[0, 5], [8, 9]]
        matches = categorize_matches_with_min_overlap(
            gsd_list_detected=self._to_interval_df(intervals_example_more_samples),
            gsd_list_reference=self._to_interval_df(interval_several_overlaps_with_example),
            overlap_threshold=5 / 9,
        )

        assert len(matches) == len(intervals_example_more_samples)
        assert_array_equal(matches.query("match_type == 'tp'")["gs_id_reference"].to_numpy(), [0])
        assert_array_equal(matches.query("match_type == 'tp'")["gs_id_detected"].to_numpy(), [0])
        assert_array_equal(matches.query("match_type == 'fp'")["gs_id_reference"].to_numpy().astype(float), [np.nan])
        assert_array_equal(matches.query("match_type == 'fp'")["gs_id_detected"].to_numpy(), [1])

    def test_several_intervals_overlap_several(self, intervals_example_more_samples):
        intervals_several_overlaps_with_example = [[0, 9], [9, 12]]  # overlapping regions are [[0, 5], [8, 9], [9, 12]]
        matches = categorize_matches_with_min_overlap(
            gsd_list_detected=self._to_interval_df(intervals_example_more_samples),
            gsd_list_reference=self._to_interval_df(intervals_several_overlaps_with_example),
            overlap_threshold=5 / 9,
        )

        assert len(matches) == len(intervals_example_more_samples)
        assert_array_equal(matches.query("match_type == 'tp'")["gs_id_reference"].to_numpy(), [0, 1])
        assert_array_equal(matches.query("match_type == 'tp'")["gs_id_detected"].to_numpy(), [0, 1])

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


class TestGeneralGsdPerformanceMetrics:
    """Tests for calculate_matched_gsd_performance_metrics method for gsd validation."""

    def test_raise_type_error_no_df(self, intervals_example):
        with pytest.raises(TypeError):
            calculate_matched_gsd_performance_metrics([1, 2, 3])

    @pytest.mark.parametrize(
        "column_names",
        [
            ["start", "end", "not_match_type"],
            ["something", "something_else", "match_type"],
            ["not_even", "enough_columns"],
        ],
    )
    def test_raise_error_on_invalid_columns(self, column_names):
        with pytest.raises(ValueError):
            calculate_matched_gsd_performance_metrics(pd.DataFrame(columns=column_names))

    def test_raise_error_on_invalid_match_type(self):
        with pytest.raises(ValueError):
            calculate_matched_gsd_performance_metrics(
                pd.DataFrame(
                    {
                        "start": [0, 1, 2, 3, 4],
                        "end": [0, 1, 3, 2, 4],
                        "match_type": ["tp", "tp", "fn", "tn", "not_valid"],
                    }
                )
            )

    def test_output(self, snapshot):
        categorized_intervals = pd.DataFrame(
            [[1, 2, "tp"], [3, 4, "fp"], [7, 8, "tn"], [10, 12, "fn"]], columns=["start", "end", "match_type"]
        )
        metrics = calculate_matched_gsd_performance_metrics(categorized_intervals)
        snapshot.assert_match(pd.DataFrame(metrics, index=[0]), "metrics")

    def test_output_no_matches(self, snapshot):
        categorized_intervals = pd.DataFrame(
            [[1, 2, "fp"], [3, 4, "fp"], [5, 6, "tn"]], columns=["start", "end", "match_type"]
        )
        metrics = calculate_matched_gsd_performance_metrics(categorized_intervals)
        snapshot.assert_match(pd.DataFrame(metrics, index=[0]), "metrics_no_match")


class TestMobilisedGsdPerformanceMetrics:
    """Tests for calculate_unmatched_gsd_performance_metrics method for gsd validation."""

    def test_raise_implausible_sampling_rate(self):
        data = pd.DataFrame([[0, 10], [15, 25]], columns=["start", "end"])
        with pytest.raises(ValueError):
            calculate_unmatched_gsd_performance_metrics(
                gsd_list_detected=data, gsd_list_reference=data, sampling_rate_hz=0
            )
        with pytest.raises(ValueError):
            calculate_unmatched_gsd_performance_metrics(
                gsd_list_detected=data, gsd_list_reference=data, sampling_rate_hz=-1
            )

    def test_output(self, snapshot):
        reference = pd.DataFrame([[0, 10], [15, 25]], columns=["start", "end"])
        detected = pd.DataFrame([[0, 10], [20, 30]], columns=["start", "end"])
        metrics = calculate_unmatched_gsd_performance_metrics(
            gsd_list_detected=detected, gsd_list_reference=reference, sampling_rate_hz=10
        )
        snapshot.assert_match(pd.DataFrame(metrics, index=[0]), "metrics")

    def test_output_no_matches(self, snapshot):
        reference = pd.DataFrame([[0, 10], [15, 25]], columns=["start", "end"])
        detected = pd.DataFrame([[10, 15], [30, 35]], columns=["start", "end"])
        metrics = calculate_unmatched_gsd_performance_metrics(
            gsd_list_detected=detected, gsd_list_reference=reference, sampling_rate_hz=10
        )
        snapshot.assert_match(pd.DataFrame(metrics, index=[0]), "metrics_no_match")

    def test_raise_zero_division_no_reference(self):
        reference = pd.DataFrame(columns=["start", "end"])
        detected = pd.DataFrame([[10, 15], [30, 35]], columns=["start", "end"])
        with pytest.raises(ZeroDivisionError):
            calculate_unmatched_gsd_performance_metrics(
                gsd_list_detected=detected,
                gsd_list_reference=reference,
                sampling_rate_hz=10,
                zero_division_hint="raise",
            )

    def test_warning_zero_division_no_reference(self):
        reference = pd.DataFrame(columns=["start", "end"])
        detected = pd.DataFrame([[10, 15], [30, 35]], columns=["start", "end"])
        with pytest.warns(UserWarning):
            calculate_unmatched_gsd_performance_metrics(
                gsd_list_detected=detected, gsd_list_reference=reference, sampling_rate_hz=10, zero_division_hint="warn"
            )

    def test_raise_invalid_zero_division_hint(self):
        reference = pd.DataFrame(columns=["start", "end"])
        with pytest.raises(ValueError):
            calculate_unmatched_gsd_performance_metrics(
                gsd_list_detected=reference,
                gsd_list_reference=reference,
                sampling_rate_hz=10,
                zero_division_hint="invalid",
            )

    def test_output_no_reference(self, snapshot):
        reference = pd.DataFrame(columns=["start", "end"])
        detected = pd.DataFrame([[10, 15], [30, 35]], columns=["start", "end"])
        metrics = calculate_unmatched_gsd_performance_metrics(
            gsd_list_detected=detected, gsd_list_reference=reference, sampling_rate_hz=10
        )
        snapshot.assert_match(pd.DataFrame(metrics, index=[0]), "metrics_no_reference")


class TestCombineDetectedReference:
    @pytest.mark.parametrize(
        "det_df, ref_df",
        [(pd.DataFrame(), "dmo_df"), ("dmo_df", pd.DataFrame()), (pd.DataFrame(), pd.DataFrame())],
    )
    def test_combine_det_ref_no_matching_empty_df(self, ref_df, det_df, request):
        ref_df = request.getfixturevalue(ref_df) if isinstance(ref_df, str) else ref_df
        det_df = request.getfixturevalue(det_df) if isinstance(det_df, str) else det_df
        with pytest.raises(ValueError):
            combine_det_with_ref_without_matching(metrics_detected=det_df, metrics_reference=ref_df)

    def test_combine_det_ref_no_matching_index_not_unique(self, dmo_df):
        df = dmo_df.copy()
        df.index = ["a"] * len(df)
        with pytest.raises(ValueError):
            combine_det_with_ref_without_matching(metrics_detected=df, metrics_reference=df)
        with pytest.raises(ValueError):
            combine_det_with_ref_without_matching(metrics_detected=dmo_df, metrics_reference=df)
        with pytest.raises(ValueError):
            combine_det_with_ref_without_matching(metrics_detected=df, metrics_reference=dmo_df)

    def test_combine_det_ref_no_matches(self, dmo_df):
        combined = combine_det_with_ref_without_matching(metrics_detected=dmo_df, metrics_reference=dmo_df)
        assert combined.shape[0] == len(dmo_df)
        assert combined.shape[1] == 2 * dmo_df.shape[1] + 2  # +2 for the wb_id columns
        assert_array_equal(combined["wb_id"]["detected"].to_numpy(), combined["wb_id"]["reference"].to_numpy())
        assert_array_equal(combined["wb_id"]["detected"].to_numpy(), dmo_df.reset_index()["wb_id"].to_numpy())
        assert_array_equal(combined.loc[:, (dmo_df.columns, "detected")].to_numpy(), dmo_df.to_numpy())
        assert_array_equal(combined.loc[:, (dmo_df.columns, "reference")].to_numpy(), dmo_df.to_numpy())

    def test_combine_det_ref_no_matching_without_wb_id(self, dmo_df):
        df_no_wb_id = dmo_df.rename_axis(index=["something", "else"])
        combined = combine_det_with_ref_without_matching(metrics_detected=df_no_wb_id, metrics_reference=df_no_wb_id)
        assert combined.shape[0] == len(dmo_df)
        assert combined.shape[1] == 2 * dmo_df.shape[1]
        assert_array_equal(combined.loc[:, (dmo_df.columns, "detected")].to_numpy(), dmo_df.to_numpy())
        assert_array_equal(combined.loc[:, (dmo_df.columns, "reference")].to_numpy(), dmo_df.to_numpy())

    def test_get_matching_gs_empty_df(self, dmo_df, matches_df):
        with pytest.raises(ValueError):
            get_matching_gs(metrics_detected=pd.DataFrame(), metrics_reference=dmo_df, matches=matches_df)
        with pytest.raises(ValueError):
            get_matching_gs(metrics_detected=dmo_df, metrics_reference=pd.DataFrame(), matches=matches_df)
        with pytest.raises(ValueError):
            get_matching_gs(metrics_detected=pd.DataFrame(), metrics_reference=pd.DataFrame(), matches=matches_df)

    def test_get_matching_gs_no_matches(self, dmo_df, matches_df):
        matches_df["match_type"] = "fp" * len(matches_df)
        combined = get_matching_gs(
            metrics_detected=dmo_df,
            metrics_reference=dmo_df,
            matches=pd.DataFrame(columns=matches_df.columns),
        )
        assert combined.empty
        assert_array_equal(
            list(combined.columns[:-2].to_numpy()), list(product(dmo_df.columns, ["detected", "reference"]))
        )
        assert_array_equal(list(combined.columns[-2:].to_numpy()), list(product(["wb_id"], ["detected", "reference"])))

    def test_get_matching_gs_invalid_matches(self, dmo_df, matches_df):
        with pytest.raises(TypeError):
            get_matching_gs(metrics_detected=dmo_df, metrics_reference=dmo_df, matches="wrong_type")

        matches_wrong_columns = matches_df.copy()
        matches_wrong_columns.columns = ["a", "b", "c"]
        with pytest.raises(ValueError):
            get_matching_gs(metrics_detected=dmo_df, metrics_reference=dmo_df, matches=matches_wrong_columns)

        matches_wrong_match_type = matches_df.copy()
        matches_wrong_match_type["match_type"] = "wrong"
        with pytest.raises(ValueError):
            get_matching_gs(metrics_detected=dmo_df, metrics_reference=dmo_df, matches=matches_wrong_match_type)

    def test_get_matching_gs_no_common_columns(self, dmo_df, matches_df):
        dummy_dmo_df_other_columns = dmo_df.copy()
        dummy_dmo_df_other_columns.columns = ["a", "b", "c"]
        dummy_dmo_df_other_columns.rename_axis(index=["something", "else"], inplace=True)
        with pytest.raises(ValueError):
            get_matching_gs(metrics_detected=dmo_df, metrics_reference=dummy_dmo_df_other_columns, matches=matches_df)
        with pytest.raises(ValueError):
            get_matching_gs(metrics_detected=dummy_dmo_df_other_columns, metrics_reference=dmo_df, matches=matches_df)

    def test_get_matching_gs(self, snapshot, dmo_df, matches_df):
        combined = get_matching_gs(metrics_detected=dmo_df, metrics_reference=dmo_df, matches=matches_df)
        assert combined.shape[0] == len(matches_df.query("match_type == 'tp'"))
        assert combined.shape[1] == 2 * dmo_df.shape[1] + 2
        assert_array_equal(combined.index, matches_df.query("match_type == 'tp'").index)
        assert_array_equal(
            list(combined.columns.to_numpy()),
            list(product(dmo_df.columns, ["detected", "reference"])) + [("wb_id", "detected"), ("wb_id", "reference")],
        )
        snapshot.assert_match(combined.to_numpy()[0], "combined")


class TestTransformationAggregationFunctions:
    def test_error_funcs(self):
        df = pd.DataFrame([[0, 1], [1, 2], [2, 3]], columns=["detected", "reference"])
        assert_series_equal(error(df), pd.Series([-1, -1, -1]))
        assert_series_equal(abs_error(df), pd.Series([1, 1, 1]))
        assert_series_equal(rel_error(df), pd.Series([-1, -1 / 2, -1 / 3]))
        assert_series_equal(abs_rel_error(df), pd.Series([1, 1 / 2, 1 / 3]))

    def test_error_funcs_with_zero_division(self):
        df = pd.DataFrame([[0, 1], [1, 2], [2, 3]], columns=["reference", "detected"])
        assert_series_equal(error(df), pd.Series([1, 1, 1]))
        assert_series_equal(abs_error(df), pd.Series([1, 1, 1]))
        assert_series_equal(rel_error(df, zero_division_hint=np.nan), pd.Series([np.nan, 1, 1 / 2]))
        assert_series_equal(abs_rel_error(df, zero_division_hint=np.nan), pd.Series([np.nan, 1, 1 / 2]))
        with pytest.warns(UserWarning):
            assert_series_equal(rel_error(df), pd.Series([np.nan, 1, 1 / 2]))
            assert_series_equal(abs_rel_error(df), pd.Series([np.nan, 1, 1 / 2]))
        with pytest.raises(ZeroDivisionError):
            rel_error(df, zero_division_hint="raise")
            abs_rel_error(df, zero_division_hint="raise")
        with pytest.raises(ValueError):
            rel_error(df, zero_division_hint="invalid")
            abs_rel_error(df, zero_division_hint="invalid")

    @pytest.mark.parametrize(
        "col_names", [["not_detected", "reference"], ["detected", "not_reference"], ["not_detected", "not_reference"]]
    )
    def test_error_funcs_with_wrong_columns(self, col_names):
        df = pd.DataFrame([[0, 1], [1, 2], [2, 3]], columns=["not_detected", "not_reference"])
        with pytest.raises(ValueError):
            error(df)
        with pytest.raises(ValueError):
            rel_error(df)
        with pytest.raises(ValueError):
            abs_error(df)
        with pytest.raises(ValueError):
            abs_rel_error(df)

    def test_error_funcs_with_wrong_num_columns(self):
        df = pd.DataFrame(
            [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],
            columns=[("detected", "a"), ("detected", "b"), ("reference", "a"), ("reference", "b")],
        )
        with pytest.raises(ValueError):
            error(df)
        with pytest.raises(ValueError):
            rel_error(df)
        with pytest.raises(ValueError):
            abs_error(df)
        with pytest.raises(ValueError):
            abs_rel_error(df)

    def test_agg_funcs(self):
        df_quantiles = pd.Series(np.arange(0, 11))
        assert quantiles(df_quantiles) == (0.5, 9.5)

        df_loa = pd.Series(np.arange(0, 11))
        df_loa = (df_loa - df_loa.mean()) / df_loa.std()
        assert_array_equal(loa(df_loa), [-1.96, 1.96])

        df_icc = pd.DataFrame([[0, 1], [0, 1], [0, 1]], columns=["detected", "reference"])
        icc_val, ci_95 = icc(df_icc)
        assert icc_val == -1
        assert_array_equal(ci_95, [-1, -1])

    def test_agg_funcs_with_nan(self):
        df_quantiles = pd.Series(np.arange(0, 12))
        df_quantiles.iloc[-1] = np.nan
        assert quantiles(df_quantiles) == (0.5, 9.5)

        df_loa = pd.Series(np.arange(0, 12))
        df_loa.iloc[-1] = np.nan
        df_loa = (df_loa - df_loa.mean()) / df_loa.std()
        assert_array_equal(loa(df_loa), [-1.96, 1.96])

        df_icc = pd.DataFrame([[0, 1], [0, 1], [0, 1], [np.nan, np.nan]], columns=["detected", "reference"])
        icc_val, ci_95 = icc(df_icc)
        assert icc_val == -1
        assert_array_equal(ci_95, [-1, -1])

    @pytest.mark.parametrize(
        "col_names", [["not_detected", "reference"], ["detected", "not_reference"], ["not_detected", "not_reference"]]
    )
    def test_icc_wrong_columns(self, col_names):
        df = pd.DataFrame([[0, 1], [0, 1], [0, 1]], columns=col_names)
        with pytest.raises(ValueError):
            icc(df)

    def test_icc_wrong_num_columns(self):
        df = pd.DataFrame(
            [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],
            columns=[("detected", "a"), ("detected", "b"), ("reference", "a"), ("reference", "b")],
        )
        with pytest.raises(ValueError):
            icc(df)


class TestCustomOperation:
    def test_correct_properties(self):
        co = CustomOperation(identifier="test", function=lambda x: x, column_name="test")
        assert co._TAG == "CustomOperation"
        assert hasattr(co, "identifier")
        assert hasattr(co, "function")
        assert hasattr(co, "column_name")


class TestApplyTransformations:
    def test_apply_transform_with_unnamed_fct(self, combined_det_ref_dmo_df):
        from functools import partial

        unnamed_trafo_func = partial(error)
        transformations = [(combined_det_ref_dmo_df.columns.get_level_values(0)[0], unnamed_trafo_func)]
        with pytest.raises(ValueError):
            apply_transformations(combined_det_ref_dmo_df, transformations)

    @staticmethod
    def _prepare_error_mock_functions(abs_rel_err_mock, abs_err_mock, rel_err_mock, err_mock, return_value):
        # set names to mocked functions to prevent unnamed function error
        err_mock.__name__ = "error"
        abs_err_mock.__name__ = "abs_error"
        rel_err_mock.__name__ = "rel_error"
        abs_rel_err_mock.__name__ = "abs_rel_error"

        # set return values for mocked functions
        err_mock.return_value = return_value
        abs_err_mock.return_value = return_value
        rel_err_mock.return_value = return_value
        abs_rel_err_mock.return_value = return_value

        return err_mock, abs_err_mock, rel_err_mock, abs_rel_err_mock

    @patch("mobgap.gsd.evaluation.error")
    @patch("mobgap.gsd.evaluation.rel_error")
    @patch("mobgap.gsd.evaluation.abs_error")
    @patch("mobgap.gsd.evaluation.abs_rel_error")
    def test_apply_transform_with_default_errors(
        self, abs_rel_err_mock, abs_err_mock, rel_err_mock, err_mock, combined_det_ref_dmo_df
    ):
        return_value = pd.Series([0] * len(combined_det_ref_dmo_df))
        err_mock, abs_err_mock, rel_err_mock, abs_rel_err_mock = self._prepare_error_mock_functions(
            abs_rel_err_mock, abs_err_mock, rel_err_mock, err_mock, return_value
        )

        transformations = get_default_error_transformations()
        res = apply_transformations(combined_det_ref_dmo_df, transformations)

        # expected number of calls for error functions
        # -2 because wb_id columns are not transformed
        # // 2 because two cols are transformed at once
        count = (combined_det_ref_dmo_df.shape[1] - 2) // 2

        assert res.shape[0] == len(combined_det_ref_dmo_df)
        assert res.shape[1] == 4 * count

        assert err_mock.call_count == count
        assert abs_err_mock.call_count == count
        assert rel_err_mock.call_count == count
        assert abs_rel_err_mock.call_count == count

    @patch("mobgap.gsd.evaluation.error")
    @patch("mobgap.gsd.evaluation.rel_error")
    @patch("mobgap.gsd.evaluation.abs_error")
    @patch("mobgap.gsd.evaluation.abs_rel_error")
    def test_apply_transform_with_custom_transformations(
        self, abs_rel_err_mock, abs_err_mock, rel_err_mock, err_mock, combined_det_ref_dmo_df
    ):
        return_value = pd.Series([0] * len(combined_det_ref_dmo_df))
        err_mock, abs_err_mock, rel_err_mock, abs_rel_err_mock = self._prepare_error_mock_functions(
            abs_rel_err_mock, abs_err_mock, rel_err_mock, err_mock, return_value
        )

        metric = combined_det_ref_dmo_df.columns.get_level_values(0)[0]
        transformations = [
            CustomOperation(identifier=metric, function=err_mock, column_name=(metric, "test_1")),
            CustomOperation(identifier=metric, function=abs_err_mock, column_name=(metric, "test_2")),
            CustomOperation(identifier=metric, function=rel_err_mock, column_name=(metric, "test_3")),
            CustomOperation(identifier=metric, function=abs_rel_err_mock, column_name=(metric, "test_4")),
            (metric, err_mock),
            (metric, abs_err_mock),
            (metric, rel_err_mock),
            (metric, abs_rel_err_mock),
        ]
        res = apply_transformations(combined_det_ref_dmo_df, transformations)
        assert res.shape[0] == len(combined_det_ref_dmo_df)
        assert res.shape[1] == len(transformations)

        count = 2
        assert err_mock.call_count == count
        assert abs_err_mock.call_count == count
        assert rel_err_mock.call_count == count
        assert abs_rel_err_mock.call_count == count

    @pytest.mark.parametrize("incompatible_trafo_func", [lambda x: x, lambda x: np.zeros(len(x))])
    def test_apply_transform_with_incompatible_transformations(self, combined_det_ref_dmo_df, incompatible_trafo_func):
        metric = combined_det_ref_dmo_df.columns.get_level_values(0)[0]
        transformations = [
            (metric, error),
            CustomOperation(identifier=metric, function=incompatible_trafo_func, column_name=(metric, "test")),
        ]
        with pytest.raises(ValueError):
            apply_transformations(combined_det_ref_dmo_df, transformations)

    def test_apply_transform_with_empty_df(self):
        df = pd.DataFrame()
        transformations = get_default_error_transformations()
        with pytest.raises(ValueError):
            apply_transformations(df, transformations)


class TestApplyAggregations:
    def test_apply_agg_with_unnamed_fct(self, combined_det_ref_dmo_df):
        from functools import partial

        unnamed_trafo_func = partial(loa)
        aggs = [
            CustomOperation(
                identifier=(combined_det_ref_dmo_df.columns.get_level_values(0)[0], "reference"),
                function=unnamed_trafo_func,
                column_name=("test_a", "test_b"),
            )
        ]
        with pytest.raises(ValueError):
            print(apply_aggregations(combined_det_ref_dmo_df, aggs))

    @staticmethod
    def _prepare_aggregation_mock_function(mock_fct, name, return_value):
        # set names to mocked functions to prevent unnamed function error
        mock_fct.__name__ = name

        # set return values for mocked functions
        mock_fct.return_value = return_value

        return mock_fct

    def test_default_aggregations(self, combined_det_ref_dmo_df_with_errors):
        aggs = get_default_aggregations()
        res = apply_aggregations(combined_det_ref_dmo_df_with_errors, aggs)

        count_metrics = combined_det_ref_dmo_df_with_errors.columns.get_level_values(0).nunique() - 1
        # for all origins
        count_mean = count_metrics * 6
        # for "detected", "reference", "abs_error", "abs_rel_error"
        count_quantiles = count_metrics * 4
        # for each "detected", "reference" pair
        count_icc = count_metrics
        # for "error", "rel_error"
        count_loa = count_metrics * 2

        assert isinstance(res, pd.Series)
        assert len(res) == count_quantiles + count_icc + count_loa + count_mean
        assert res.index.get_level_values(0).nunique() == 4  # function names mean, icc, quantiles, loa
        assert res.index.get_level_values(1).nunique() == count_metrics
        assert res.index.get_level_values(2).nunique() == 7  # all origins + "all"

    @pytest.mark.parametrize(
        ("identifier", "input_df"),
        [
            ("single_col", pd.DataFrame(np.zeros((10, 1)), columns=["single_col"])),
            (("single_col",), pd.DataFrame(np.zeros((10, 1)), columns=["single_col"])),
            (
                ("multi_col", "multi_col_2"),
                pd.DataFrame(np.zeros((10, 1)), columns=pd.MultiIndex.from_tuples([("multi_col", "multi_col_2")])),
            ),
            (
                ("multi_col", "multi_col_2", "multi_col_3"),
                pd.DataFrame(
                    np.zeros((10, 1)), columns=pd.MultiIndex.from_tuples([("multi_col", "multi_col_2", "multi_col_3")])
                ),
            ),
        ],
    )
    def test_agg_aggregations(self, identifier, input_df):
        functions = ["mean", "std"]
        aggregations = [(identifier, functions)]
        print(input_df)
        res = apply_aggregations(input_df, aggregations)

        assert isinstance(res, pd.Series)
        assert len(res) == len(functions)
        assert res.index.nlevels == (len(identifier) + 1) if isinstance(identifier, tuple) else 2
        print(res)

    @pytest.mark.parametrize("invalid_id", [(1,), (2, 3, 4), ["test", "test"]])
    def test_agg_aggregations_with_wrong_identifier(self, invalid_id, combined_det_ref_dmo_df_with_errors):
        functions = ["mean", "std"]
        aggregations = [(invalid_id, functions)]
        with pytest.raises(ValueError):
            apply_aggregations(combined_det_ref_dmo_df_with_errors, aggregations)

    @patch("mobgap.gsd.evaluation.loa")
    @patch("mobgap.gsd.evaluation.icc")
    @patch("mobgap.gsd.evaluation.quantiles")
    def test_apply_custom_aggregations(self, quantiles_mock, icc_mock, loa_mock, combined_det_ref_dmo_df_with_errors):
        quantiles_mock = self._prepare_aggregation_mock_function(quantiles_mock, "quantiles", 0)
        icc_mock = self._prepare_aggregation_mock_function(icc_mock, "icc", 0)
        loa_mock = self._prepare_aggregation_mock_function(loa_mock, "loa", 0)

        metrics = combined_det_ref_dmo_df_with_errors.columns.get_level_values(0).unique()
        origins = combined_det_ref_dmo_df_with_errors.columns.get_level_values(1).unique()

        aggregations = [
            *[
                CustomOperation(identifier=(metric, origin), function=quantiles_mock, column_name=(metric, origin))
                for metric in metrics
                for origin in origins
            ],
            *[
                CustomOperation(identifier=(metric, origin), function=loa_mock, column_name=(metric, origin))
                for metric in metrics
                for origin in origins
            ],
            *[CustomOperation(identifier=metric, function=icc_mock, column_name=(metric, "all")) for metric in metrics],
        ]

        res = apply_aggregations(combined_det_ref_dmo_df_with_errors, aggregations)

        assert isinstance(res, pd.Series)
        # for all 6 origins: add calculation of built-in "mean" to the expected count
        assert len(res) == len(aggregations)

        assert loa_mock.call_count == len(metrics) * len(origins)
        assert icc_mock.call_count == len(metrics)
        assert quantiles_mock.call_count == len(metrics) * len(origins)

    @patch("mobgap.gsd.evaluation.loa")
    @patch("mobgap.gsd.evaluation.quantiles")
    def test_apply_custom_agg_with_nan_results(self, quantiles_mock, loa_mock, combined_det_ref_dmo_df_with_errors):
        # TODO: fix this
        quantiles_mock = self._prepare_aggregation_mock_function(quantiles_mock, "quantiles", 0)
        loa_mock = self._prepare_aggregation_mock_function(loa_mock, "loa", np.nan)

        metrics = combined_det_ref_dmo_df_with_errors.columns.get_level_values(0).unique()
        origins = combined_det_ref_dmo_df_with_errors.columns.get_level_values(1).unique()

        aggregations = [
            *[
                CustomOperation(identifier=(metric, origin), function=quantiles_mock, column_name=(metric, origin))
                for metric in metrics
                for origin in origins
            ],
            *[
                CustomOperation(identifier=(metric, origin), function=loa_mock, column_name=(metric, origin))
                for metric in metrics
                for origin in origins
            ],
        ]

        res = apply_aggregations(combined_det_ref_dmo_df_with_errors, aggregations)

        assert isinstance(res, pd.Series)
        # for all 6 origins: add calculation of built-in "mean" to the expected count
        assert len(res) == len(aggregations)

        assert loa_mock.call_count == len(metrics) * len(origins)
        assert quantiles_mock.call_count == len(metrics) * len(origins)

    @pytest.mark.parametrize("incompatible_col_names", ["test", ("test", "test", "test")])
    def test_apply_agg_with_incompatible_aggregations(self, combined_det_ref_dmo_df, incompatible_col_names):
        metric = combined_det_ref_dmo_df.columns.get_level_values(0)[0]
        aggregations = [
            ((metric, "reference"), "mean"),
            CustomOperation(identifier=metric, function=lambda x: 0, column_name=incompatible_col_names),
        ]
        with pytest.raises(ValueError):
            apply_aggregations(combined_det_ref_dmo_df, aggregations)

    def test_apply_agg_with_empty_df(self):
        df = pd.DataFrame()
        aggs = get_default_aggregations()
        with pytest.raises(ValueError):
            apply_aggregations(df, aggs)
