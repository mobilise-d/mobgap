import warnings
from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from mobgap.gait_sequences.evaluation import (
    _get_tn_intervals,
    calculate_matched_gsd_performance_metrics,
    calculate_unmatched_gsd_performance_metrics,
    categorize_intervals,
    categorize_intervals_per_sample,
    get_matching_intervals,
)


@pytest.fixture
def intervals_example():
    return [[1, 3], [5, 7]]


@pytest.fixture
def intervals_example_more_samples():
    return [[0, 5], [8, 13]]


@pytest.fixture
def intervals_example_with_id():
    return pd.DataFrame([[1, 3, 0], [5, 7, 1]], columns=["start", "end", "id"]).set_index("id")


@pytest.fixture
def dmo_df():
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["metric_a", "metric_b", "metric_c"])
    df.index = pd.MultiIndex.from_tuples([("a", 1), ("a", 2)], names=["group", "wb_id"])
    return df


@pytest.fixture
def matches_df():
    return pd.DataFrame(
        {
            "gs_id_detected": [("a", 1), ("a", 2), np.nan],
            "gs_id_reference": [("a", 2), np.nan, ("a", 1)],
            "match_type": ["tp", "fp", "fn"],
        }
    )


class TestCategorizeIntervalsPerSample:
    """Tests for categorize_intervals_per_sample method for gsd validation."""

    def test_raise_type_error_no_df(self, intervals_example):
        with pytest.raises(TypeError):
            categorize_intervals_per_sample(gsd_list_detected=intervals_example, gsd_list_reference=intervals_example)

    def test_raise_value_error_wrong_input_columns(self, intervals_example):
        with pytest.raises(ValueError):
            # only default column names
            categorize_intervals_per_sample(
                gsd_list_detected=pd.DataFrame(intervals_example), gsd_list_reference=pd.DataFrame(intervals_example)
            )

    def test_raise_wrong_num_samples(self, intervals_example_with_id):
        with pytest.raises(ValueError):
            categorize_intervals_per_sample(
                gsd_list_detected=pd.DataFrame(intervals_example_with_id),
                gsd_list_reference=pd.DataFrame(intervals_example_with_id),
                n_overall_samples=2,
            )

    @pytest.mark.parametrize(
        ("detected", "reference", "tp_fp_fn_list"),
        [
            ("intervals_example", pd.DataFrame(), [[], [[1, 3], [5, 7]], []]),
            (pd.DataFrame(), "intervals_example", [[], [], [[1, 3], [5, 7]]]),
            (pd.DataFrame(), pd.DataFrame(), [[], [], []]),
        ],
    )
    def test_empty_df_as_input(self, detected, reference, tp_fp_fn_list, request):
        detected = detected if isinstance(detected, pd.DataFrame) else request.getfixturevalue(detected)
        reference = reference if isinstance(reference, pd.DataFrame) else request.getfixturevalue(reference)

        self._assert_equal_tp_fp_fn(detected, reference, *tp_fp_fn_list)

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
        result = categorize_intervals_per_sample(
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


class TestCategorizeIntervals:
    """`Tests for categorize_intervals` method for gsd validation."""

    def test_raise_type_error_no_df(self, intervals_example):
        with pytest.raises(TypeError):
            categorize_intervals(gsd_list_detected=intervals_example, gsd_list_reference=intervals_example)

    def test_raise_value_error_wrong_input_columns(self, intervals_example, intervals_example_with_id):
        with pytest.raises(ValueError):
            # only default columns
            categorize_intervals(
                gsd_list_detected=pd.DataFrame(intervals_example),
                gsd_list_reference=pd.DataFrame(intervals_example_with_id),
            )

    @pytest.mark.parametrize("min_overlap", [0, 0.5, 2])
    def test_raise_value_error_invalid_overlap(self, intervals_example_with_id, min_overlap):
        with pytest.raises(ValueError):
            categorize_intervals(
                gsd_list_detected=self._to_interval_df(intervals_example_with_id),
                gsd_list_reference=self._to_interval_df(intervals_example_with_id),
                overlap_threshold=min_overlap,
            )

    def test_raise_value_error_invalid_index(self, intervals_example_with_id):
        index_not_unique = pd.DataFrame(intervals_example_with_id)
        index_not_unique["id"] = ["id"] * len(intervals_example_with_id)
        index_not_unique = index_not_unique.set_index("id")
        with pytest.raises(ValueError):
            categorize_intervals(
                gsd_list_detected=index_not_unique, gsd_list_reference=index_not_unique, overlap_threshold=1
            )

    def test_input_multiindex_warning(self, intervals_example_with_id):
        multiindex = intervals_example_with_id.copy()
        multiindex.index = pd.MultiIndex.from_tuples([("a", 1), ("a", 2)], names=["something", "gsd_id"])
        with pytest.warns(Warning):
            categorize_intervals(gsd_list_detected=multiindex, gsd_list_reference=intervals_example_with_id)
        with pytest.warns(Warning):
            categorize_intervals(gsd_list_detected=intervals_example_with_id, gsd_list_reference=multiindex)
        with pytest.warns(Warning):
            categorize_intervals(gsd_list_detected=multiindex, gsd_list_reference=multiindex)

    def test_input_multiindex_warning_suppressed(self, intervals_example_with_id):
        multiindex = intervals_example_with_id.copy()
        multiindex.index = pd.MultiIndex.from_tuples([("a", 1), ("a", 2)], names=["something", "ic_id"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            categorize_intervals(
                gsd_list_detected=multiindex, gsd_list_reference=intervals_example_with_id, multiindex_warning=False
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            categorize_intervals(
                gsd_list_detected=intervals_example_with_id, gsd_list_reference=multiindex, multiindex_warning=False
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            categorize_intervals(gsd_list_detected=multiindex, gsd_list_reference=multiindex, multiindex_warning=False)

    @pytest.mark.parametrize(
        ("detected", "reference", "match_type"),
        [
            ("intervals_example", pd.DataFrame(), "fp"),
            (pd.DataFrame(), "intervals_example", "fn"),
            (pd.DataFrame(), pd.DataFrame(), None),
        ],
    )
    def test_empty_df_as_input(self, detected, reference, match_type, request):
        detected = detected if isinstance(detected, pd.DataFrame) else request.getfixturevalue(detected)
        reference = reference if isinstance(reference, pd.DataFrame) else request.getfixturevalue(reference)

        detected = self._to_interval_df(detected)
        reference = self._to_interval_df(reference)

        matches = categorize_intervals(gsd_list_detected=detected, gsd_list_reference=reference)
        assert len(matches) == max(len(detected), len(reference))
        if match_type:
            assert len(matches["match_type"].unique()) == 1
            assert matches["match_type"].unique()[0] == match_type
        else:
            assert matches.empty
            assert matches.columns.tolist() == ["gs_id_detected", "gs_id_reference", "match_type"]

    def test_validation_all_tp(self, intervals_example):
        matches = categorize_intervals(
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
        matches = categorize_intervals(gsd_list_detected=ic_list_multiindex, gsd_list_reference=ic_list_multiindex)
        assert np.all(matches["match_type"] == "tp")
        assert len(matches) == len(ic_list_multiindex)
        assert_array_equal(matches["gs_id_detected"].to_numpy(), ic_list_multiindex.index.to_flat_index())
        assert_array_equal(matches["gs_id_reference"].to_numpy(), ic_list_multiindex.index.to_flat_index())

    def test_validation_all_tp_with_id(self, intervals_example_with_id):
        ref = intervals_example_with_id.copy()
        ref["id"] = ["ref"] * len(ref)
        matches = categorize_intervals(
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
        matches = categorize_intervals(
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
        matches = categorize_intervals(
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
        matches = categorize_intervals(
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

        matches = categorize_intervals(
            gsd_list_detected=self._to_interval_df(intervals_example_more_samples),
            gsd_list_reference=self._to_interval_df(intervals_partial_overlap_with_example),
            overlap_threshold=0.6,
        )

        assert len(matches) == len(intervals_example_more_samples)
        assert_array_equal(matches.query("match_type == 'tp'")["gs_id_reference"].to_numpy(), [0, 1])
        assert_array_equal(matches.query("match_type == 'tp'")["gs_id_detected"].to_numpy(), [0, 1])

    def test_partial_overlap_start_or_end(self, intervals_example_more_samples):
        intervals_partial_overlap_with_example = [[2, 5], [8, 11]]  # overlapping regions are [[2, 5], [8, 11]]

        matches = categorize_intervals(
            gsd_list_detected=self._to_interval_df(intervals_example_more_samples),
            gsd_list_reference=self._to_interval_df(intervals_partial_overlap_with_example),
            overlap_threshold=0.6,
        )

        assert len(matches) == len(intervals_example_more_samples)
        assert_array_equal(matches.query("match_type == 'tp'")["gs_id_reference"].to_numpy(), [0, 1])
        assert_array_equal(matches.query("match_type == 'tp'")["gs_id_detected"].to_numpy(), [0, 1])

    def test_several_intervals_overlap_one(self, intervals_example_more_samples):
        interval_several_overlaps_with_example = [[0, 9]]  # overlapping regions are [[0, 5], [8, 9]]
        matches = categorize_intervals(
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
        matches = categorize_intervals(
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
    def test_get_matching_gs_empty_df(self, dmo_df, matches_df):
        with pytest.raises(ValueError):
            get_matching_intervals(metrics_detected=pd.DataFrame(), metrics_reference=dmo_df, matches=matches_df)
        with pytest.raises(ValueError):
            get_matching_intervals(metrics_detected=dmo_df, metrics_reference=pd.DataFrame(), matches=matches_df)
        with pytest.raises(ValueError):
            get_matching_intervals(
                metrics_detected=pd.DataFrame(), metrics_reference=pd.DataFrame(), matches=matches_df
            )

    def test_get_matching_gs_no_matches(self, dmo_df, matches_df):
        matches_df["match_type"] = "fp"
        combined = get_matching_intervals(
            metrics_detected=dmo_df,
            metrics_reference=dmo_df,
            matches=pd.DataFrame(columns=matches_df.columns),
        )
        assert combined.empty
        cols = np.array(combined.columns.to_list())
        cols.sort()
        expected_cols = np.array(list(product([*dmo_df.columns, "orig_index"], ["detected", "reference"])))
        expected_cols.sort()
        assert_array_equal(cols, expected_cols)

    def test_get_matching_gs_invalid_matches(self, dmo_df, matches_df):
        with pytest.raises(TypeError):
            get_matching_intervals(metrics_detected=dmo_df, metrics_reference=dmo_df, matches="wrong_type")

        matches_wrong_columns = matches_df.copy()
        matches_wrong_columns.columns = ["a", "b", "c"]
        with pytest.raises(ValueError):
            get_matching_intervals(metrics_detected=dmo_df, metrics_reference=dmo_df, matches=matches_wrong_columns)

        matches_wrong_match_type = matches_df.copy()
        matches_wrong_match_type["match_type"] = "wrong"
        with pytest.raises(ValueError):
            get_matching_intervals(metrics_detected=dmo_df, metrics_reference=dmo_df, matches=matches_wrong_match_type)

    def test_get_matching_gs_no_common_columns(self, dmo_df, matches_df):
        dummy_dmo_df_other_columns = dmo_df.copy()
        dummy_dmo_df_other_columns.columns = ["a", "b", "c"]
        dummy_dmo_df_other_columns.rename_axis(index=["something", "else"], inplace=True)
        with pytest.raises(ValueError):
            get_matching_intervals(
                metrics_detected=dmo_df, metrics_reference=dummy_dmo_df_other_columns, matches=matches_df
            )
        with pytest.raises(ValueError):
            get_matching_intervals(
                metrics_detected=dummy_dmo_df_other_columns, metrics_reference=dmo_df, matches=matches_df
            )

    def test_get_matching_gs(self, snapshot, dmo_df, matches_df):
        combined = get_matching_intervals(metrics_detected=dmo_df, metrics_reference=dmo_df, matches=matches_df)
        assert combined.shape[0] == len(matches_df.query("match_type == 'tp'"))
        assert combined.shape[1] == 2 * dmo_df.shape[1] + 2
        assert_array_equal(combined.index, matches_df.query("match_type == 'tp'").index)
        assert_array_equal(
            list(combined.columns.to_numpy()),
            list(product(dmo_df.columns, ["detected", "reference"]))
            + [("orig_index", "detected"), ("orig_index", "reference")],
        )
        # We combine the columns for the snapshot test
        combined.columns = ["_".join(col) for col in combined.columns]
        snapshot.assert_match(combined, "combined")
