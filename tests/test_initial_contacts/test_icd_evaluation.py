import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from mobgap.initial_contacts.evaluation import (
    _match_label_lists,
    calculate_matched_icd_performance_metrics,
    categorize_ic_list,
)


@pytest.fixture
def create_ic_list_default():
    return pd.DataFrame([10, 20, 40, 50], columns=["ic"]).rename_axis("ic_id")


class TestMatchLabelLists:
    def test_empty_input(self, create_ic_list_default):
        matches_left, matches_right = _match_label_lists([], create_ic_list_default.to_numpy())
        assert matches_left.size == 0
        assert matches_right.size == 0

        matches_left, matches_right = _match_label_lists(create_ic_list_default.to_numpy(), [])
        assert matches_left.size == 0
        assert matches_right.size == 0

    def test_too_many_dimensions_error(self):
        with pytest.raises(ValueError):
            invalid_input = np.array([[[1, 2], [2, 3]]])
            _match_label_lists(invalid_input, invalid_input)

    def test_perfect_match(self, create_ic_list_default):
        matches_left, matches_right = _match_label_lists(
            create_ic_list_default.to_numpy(), create_ic_list_default.to_numpy()
        )
        assert_array_equal(matches_left, [0, 1, 2, 3])
        assert_array_equal(matches_right, [0, 1, 2, 3])

    @pytest.mark.parametrize("tolerance", [[0, 0, 0, 1], [1, 1, 0, 0], [1, 3, 1, 3], [3, 3, 3, 3]])
    def test_all_within_tolerance(self, create_ic_list_default, tolerance):
        ic_list_with_tolerance = pd.DataFrame(create_ic_list_default["ic"] + tolerance)
        matches_left, matches_right = _match_label_lists(
            create_ic_list_default.to_numpy(), ic_list_with_tolerance.to_numpy(), tolerance_samples=max(tolerance)
        )
        assert_array_equal(matches_left, [0, 1, 2, 3])
        assert_array_equal(matches_right, [0, 1, 2, 3])

        matches_left, matches_right = _match_label_lists(
            create_ic_list_default.to_numpy(), ic_list_with_tolerance.to_numpy(), tolerance_samples=max(tolerance)
        )
        assert_array_equal(matches_left, [0, 1, 2, 3])
        assert_array_equal(matches_right, [0, 1, 2, 3])

    def test_no_match(self, create_ic_list_default):
        ic_list_with_tolerance = pd.DataFrame(create_ic_list_default["ic"] + 5)
        matches_left, matches_right = _match_label_lists(
            create_ic_list_default.to_numpy(), ic_list_with_tolerance, tolerance_samples=4
        )
        assert_array_equal(matches_left, [])
        assert_array_equal(matches_right, [])

        matches_left, matches_right = _match_label_lists(
            ic_list_with_tolerance, create_ic_list_default, tolerance_samples=4
        )
        assert_array_equal(matches_left, [])
        assert_array_equal(matches_right, [])

    def test_equal_distance_between_two(self, create_ic_list_default):
        ic_list_with_equal_distance = np.array([15, 30, 45])
        matches_left, matches_right = _match_label_lists(
            create_ic_list_default.to_numpy(), ic_list_with_equal_distance, tolerance_samples=10
        )
        assert_array_equal(matches_left, [0, 2])
        assert_array_equal(matches_right, [0, 2])

    def test_one_to_one(self, create_ic_list_default):
        ic_list_with_equal_distance = np.array([18, 22])
        matches_left, matches_right = _match_label_lists(
            create_ic_list_default.to_numpy(), ic_list_with_equal_distance, tolerance_samples=2
        )
        assert_array_equal(matches_left, [1])
        assert_array_equal(matches_right, [0])

        matches_left, matches_right = _match_label_lists(
            ic_list_with_equal_distance, create_ic_list_default.to_numpy(), tolerance_samples=2
        )
        assert_array_equal(matches_left, [0])
        assert_array_equal(matches_right, [1])

    @pytest.mark.parametrize(
        "ic_list_left, ic_list_right, tolerance, output_left, output_right",
        [
            ([10, 20, 30, 40], [22, 35, 50], 5, [1, 2], [0, 1]),
            ([10, 20, 30, 40], [22, 35, 50], 10, [1, 2], [0, 1]),
            ([10, 20, 30, 40], [15, 46], 5, [0], [0]),
            ([10, 20, 30, 40], [15, 46], 10, [0, 3], [0, 1]),
            ([10, 20, 30, 40], [12, 23, 28, 33], 2, [0, 2], [0, 2]),
        ],
    )
    def test_mixed_examples(self, ic_list_left, ic_list_right, tolerance, output_left, output_right):
        matches_left, matches_right = _match_label_lists(
            np.array(ic_list_left), np.array(ic_list_right), tolerance_samples=tolerance
        )
        assert_array_equal(matches_left, output_left)
        assert_array_equal(matches_right, output_right)


class TestEvaluateInitialContactList:
    @staticmethod
    def _create_ic_list(data):
        df = pd.DataFrame(data, columns=["ic"])
        df.index.name = "ic_id"
        return df

    def test_invalid_tolerance_error(self, create_ic_list_default):
        with pytest.raises(ValueError):
            categorize_ic_list(
                ic_list_detected=create_ic_list_default, ic_list_reference=create_ic_list_default, tolerance_samples=-1
            )

    def test_input_no_df_error(self):
        with pytest.raises(TypeError):
            categorize_ic_list(ic_list_detected=[1], ic_list_reference=[1])

    def test_input_wrong_cols_error(self, create_ic_list_default):
        wrong_cols = pd.DataFrame([1, 2, 3, 4], columns=["wrong"])
        with pytest.raises(ValueError):
            categorize_ic_list(ic_list_detected=create_ic_list_default, ic_list_reference=wrong_cols)
        with pytest.raises(ValueError):
            categorize_ic_list(ic_list_detected=wrong_cols, ic_list_reference=create_ic_list_default)

    def test_input_index_not_unique(self, create_ic_list_default):
        wrong_index = pd.DataFrame([1, 2, 3, 4], columns=["ic"], index=[1, 1, 2, 3])
        with pytest.raises(ValueError):
            categorize_ic_list(ic_list_detected=create_ic_list_default, ic_list_reference=wrong_index)
        with pytest.raises(ValueError):
            categorize_ic_list(ic_list_detected=wrong_index, ic_list_reference=create_ic_list_default)

    def test_input_multiindex_not_unique(self, create_ic_list_default):
        multiindex_ic_list = create_ic_list_default.copy()
        multiindex_ic_list.index = pd.MultiIndex.from_tuples(
            [("a", 1), ("a", 1), ("b", 3), ("b", 4)], names=["level_a", "level_b"]
        )
        with pytest.raises(ValueError):
            categorize_ic_list(ic_list_detected=multiindex_ic_list, ic_list_reference=create_ic_list_default)
        with pytest.raises(ValueError):
            categorize_ic_list(ic_list_detected=create_ic_list_default, ic_list_reference=multiindex_ic_list)
        with pytest.raises(ValueError):
            categorize_ic_list(ic_list_detected=multiindex_ic_list, ic_list_reference=multiindex_ic_list)

    def test_input_multiindex_warning(self, create_ic_list_default):
        multiindex = create_ic_list_default.copy()
        multiindex.index = pd.MultiIndex.from_tuples(
            [("a", 1), ("a", 2), ("b", 3), ("b", 4)], names=["something", "ic_id"]
        )
        with pytest.warns(Warning):
            categorize_ic_list(ic_list_detected=multiindex, ic_list_reference=create_ic_list_default)
        with pytest.warns(Warning):
            categorize_ic_list(ic_list_detected=create_ic_list_default, ic_list_reference=multiindex)
        with pytest.warns(Warning):
            categorize_ic_list(ic_list_detected=multiindex, ic_list_reference=multiindex)

    def test_input_multiindex_warning_suppressed(self, create_ic_list_default):
        multiindex = create_ic_list_default.copy()
        multiindex.index = pd.MultiIndex.from_tuples(
            [("a", 1), ("a", 2), ("b", 3), ("b", 4)], names=["something", "ic_id"]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            categorize_ic_list(
                ic_list_detected=multiindex, ic_list_reference=create_ic_list_default, multiindex_warning=False
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            categorize_ic_list(
                ic_list_detected=create_ic_list_default, ic_list_reference=multiindex, multiindex_warning=False
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            categorize_ic_list(ic_list_detected=multiindex, ic_list_reference=multiindex, multiindex_warning=False)

    @pytest.mark.parametrize(
        ("detected", "reference", "match_type"),
        [
            ("create_ic_list_default", pd.DataFrame(), "fp"),
            (pd.DataFrame(), "create_ic_list_default", "fn"),
            (pd.DataFrame(), pd.DataFrame(), None),
        ],
    )
    def test_empty_df_as_input(self, detected, reference, match_type, request):
        detected = (
            pd.DataFrame(detected, columns=["ic"]).rename_axis("ic_id")
            if isinstance(detected, pd.DataFrame)
            else request.getfixturevalue(detected)
        )

        reference = (
            pd.DataFrame(reference, columns=["ic"]).rename_axis("ic_id")
            if isinstance(reference, pd.DataFrame)
            else request.getfixturevalue(reference)
        )

        matches = categorize_ic_list(ic_list_detected=detected, ic_list_reference=reference)

        assert len(matches) == max(len(detected), len(reference))
        if match_type:
            assert len(matches["match_type"].unique()) == 1
            assert matches["match_type"].unique()[0] == match_type
        else:
            assert matches.empty
            assert matches.columns.tolist() == ["ic_id_detected", "ic_id_reference", "match_type"]

    def test_perfect_match(self, create_ic_list_default):
        matches = categorize_ic_list(ic_list_detected=create_ic_list_default, ic_list_reference=create_ic_list_default)
        assert np.all(matches["match_type"] == "tp")
        assert len(matches["match_type"] == "tp") == len(create_ic_list_default)

    def test_empty_reference(self, create_ic_list_default):
        reference = self._create_ic_list([])
        matches = categorize_ic_list(ic_list_detected=create_ic_list_default, ic_list_reference=reference)
        assert np.all(matches["match_type"] == "fp")
        assert len(matches["match_type"] == "fp") == len(create_ic_list_default)

    def test_empty_detected(self, create_ic_list_default):
        detected = self._create_ic_list([])
        matches = categorize_ic_list(ic_list_detected=detected, ic_list_reference=create_ic_list_default)
        assert np.all(matches["match_type"] == "fn")
        assert len(matches["match_type"] == "fn") == len(create_ic_list_default)

    def test_with_matches(self, create_ic_list_default):
        list_detected = self._create_ic_list([11, 22, 39, 45])
        matches = categorize_ic_list(
            ic_list_detected=list_detected, ic_list_reference=create_ic_list_default, tolerance_samples=1
        )
        assert_array_equal(matches.query("match_type == 'tp'")["ic_id_detected"].to_numpy(), [0, 2])
        assert_array_equal(matches.query("match_type == 'tp'")["ic_id_reference"].to_numpy(), [0, 2])

        assert_array_equal(matches.query("match_type == 'fp'")["ic_id_detected"].to_numpy(), [1, 3])
        assert_array_equal(
            matches.query("match_type == 'fp'")["ic_id_reference"].to_numpy().astype(float), [np.nan, np.nan]
        )

        assert_array_equal(
            matches.query("match_type == 'fn'")["ic_id_detected"].to_numpy().astype(float), [np.nan, np.nan]
        )
        assert_array_equal(matches.query("match_type == 'fn'")["ic_id_reference"].to_numpy(), [1, 3])

        assert len(create_ic_list_default) == len(matches.query("match_type == 'tp'")) + len(
            matches.query("match_type == 'fn'")
        )

    def test_input_range_index(self, create_ic_list_default):
        ic_list_range_index = create_ic_list_default.reset_index()
        matches = categorize_ic_list(ic_list_detected=ic_list_range_index, ic_list_reference=ic_list_range_index)
        assert np.all(matches["match_type"] == "tp")
        assert len(matches["match_type"] == "tp") == len(create_ic_list_default)
        assert_array_equal(matches.columns.to_numpy(), ["ic_id_detected", "ic_id_reference", "match_type"])
        assert matches.index.name is None
        assert_array_equal(matches.index.to_numpy(), [0, 1, 2, 3])

    def test_input_multiindex(self, create_ic_list_default):
        ic_list_multiindex = create_ic_list_default.copy()
        ic_list_multiindex.index = pd.MultiIndex.from_tuples(
            [("a", 1), ("a", 2), ("b", 3), ("b", 4)], names=["something", "something_else"]
        )
        matches = categorize_ic_list(ic_list_detected=ic_list_multiindex, ic_list_reference=ic_list_multiindex)
        assert np.all(matches["match_type"] == "tp")
        assert len(matches["match_type"] == "tp") == len(create_ic_list_default)
        assert_array_equal(matches.columns.to_numpy(), ["ic_id_detected", "ic_id_reference", "match_type"])
        assert matches.index.name is None
        assert_array_equal(matches.index.to_numpy(), [0, 1, 2, 3])
        assert_array_equal(ic_list_multiindex.index.to_flat_index(), matches["ic_id_detected"].to_numpy())
        assert_array_equal(ic_list_multiindex.index.to_flat_index(), matches["ic_id_reference"].to_numpy())

    def test_segmented_stride_list_no_match(self, create_ic_list_default):
        detected = self._create_ic_list([15, 15, 35, 45])
        matches = categorize_ic_list(
            ic_list_detected=detected, ic_list_reference=create_ic_list_default, tolerance_samples=4
        )
        assert matches.query("match_type == 'tp'").empty

        assert_array_equal(
            matches.query("match_type == 'fn'")["ic_id_detected"].to_numpy().astype(float),
            [np.nan, np.nan, np.nan, np.nan],
        )
        assert_array_equal(matches.query("match_type == 'fn'")["ic_id_reference"].to_numpy(), [0, 1, 2, 3])

        assert_array_equal(matches.query("match_type == 'fp'")["ic_id_detected"].to_numpy(), [0, 1, 2, 3])
        assert_array_equal(
            matches.query("match_type == 'fp'")["ic_id_reference"].to_numpy().astype(float),
            [np.nan, np.nan, np.nan, np.nan],
        )

        assert len(create_ic_list_default) == len(matches.query("match_type == 'tp'")) + len(
            matches.query("match_type == 'fn'")
        )

    def test_double_match_detected(self):
        reference = self._create_ic_list([20])
        detected = self._create_ic_list([18, 22])

        matches = categorize_ic_list(ic_list_detected=detected, ic_list_reference=reference, tolerance_samples=2)

        assert_array_equal(matches.query("match_type == 'tp'")["ic_id_detected"].to_numpy(), [0])
        assert_array_equal(matches.query("match_type == 'tp'")["ic_id_reference"].to_numpy(), [0])

        assert_array_equal(matches.query("match_type == 'fp'")["ic_id_detected"].to_numpy(), [1])
        assert_array_equal(matches.query("match_type == 'fp'")["ic_id_reference"].to_numpy().astype(float), [np.nan])

        assert matches.query("match_type == 'fn'").empty

        assert len(reference) == len(matches.query("match_type == 'tp'"))


class TestCalculateIcdMetrics:
    def test_empty_input(self):
        matches = pd.DataFrame(columns=["ic_id_detected", "ic_id_reference", "match_type"])
        metrics = calculate_matched_icd_performance_metrics(matches)
        for val in metrics.values():
            assert val == 0

    def test_ic_list_input(self, snapshot):
        matches = pd.DataFrame(
            {
                "ic_id_detected": [0, 1, 2, np.nan, 3],
                "ic_id_reference": [0, 1, 3, 2, np.nan],
                "match_type": ["tp", "tp", "tp", "fn", "fp"],
            }
        )
        metrics = calculate_matched_icd_performance_metrics(matches)
        snapshot.assert_match(pd.DataFrame(metrics, index=[0]), "metrics")

    def test_raise_error_on_invalid_input(self):
        with pytest.raises(TypeError):
            calculate_matched_icd_performance_metrics([1, 2, 3])

    @pytest.mark.parametrize(
        "column_names",
        [
            ["ic_id_detected", "ic_id_reference", "not_match_type"],
            ["something", "something_else", "match_type"],
            ["not_even", "enough_columns"],
        ],
    )
    def test_raise_error_on_invalid_columns(self, column_names):
        with pytest.raises(ValueError):
            calculate_matched_icd_performance_metrics(pd.DataFrame(columns=column_names))

    def test_raise_error_on_invalid_match_type(self):
        with pytest.raises(ValueError):
            calculate_matched_icd_performance_metrics(
                pd.DataFrame(
                    {
                        "ic_id_detected": [0, 1, 2, 3, 4],
                        "ic_id_reference": [0, 1, 3, 2, 4],
                        "match_type": ["tp", "tp", "tp", "fn", "not_valid"],
                    }
                )
            )
