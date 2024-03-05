import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from gaitlink.icd.evaluation import _match_label_lists, calculate_icd_performance_metrics, evaluate_ic_list


@pytest.fixture()
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
            evaluate_ic_list(create_ic_list_default, create_ic_list_default, tolerance_samples=-1)

    def test_input_no_df_error(self):
        with pytest.raises(TypeError):
            evaluate_ic_list([1], [1])

    def test_input_wrong_cols_error(self, create_ic_list_default):
        wrong_cols = pd.DataFrame([1, 2, 3, 4], columns=["wrong"])
        with pytest.raises(ValueError):
            evaluate_ic_list(create_ic_list_default, wrong_cols)
        with pytest.raises(ValueError):
            evaluate_ic_list(wrong_cols, create_ic_list_default)

    def test_input_index_not_unique(self, create_ic_list_default):
        wrong_index = pd.DataFrame([1, 2, 3, 4], columns=["ic"], index=[1, 1, 2, 3])
        with pytest.raises(ValueError):
            evaluate_ic_list(create_ic_list_default, wrong_index)
        with pytest.raises(ValueError):
            evaluate_ic_list(wrong_index, create_ic_list_default)

    def test_input_multiindex_warning(self, create_ic_list_default):
        multiindex = create_ic_list_default.copy()
        multiindex.index = pd.MultiIndex.from_tuples(
            [("a", 1), ("a", 2), ("b", 3), ("b", 4)], names=["something", "ic_id"]
        )
        with pytest.warns(Warning):
            evaluate_ic_list(multiindex, create_ic_list_default)
        with pytest.warns(Warning):
            evaluate_ic_list(create_ic_list_default, multiindex)
        with pytest.warns(Warning):
            evaluate_ic_list(multiindex, multiindex)

    def test_input_multiindex_warning_suppressed(self, create_ic_list_default):
        multiindex = create_ic_list_default.copy()
        multiindex.index = pd.MultiIndex.from_tuples(
            [("a", 1), ("a", 2), ("b", 3), ("b", 4)], names=["something", "ic_id"]
        )
        with pytest.warns(None) as record:
            evaluate_ic_list(multiindex, create_ic_list_default, multiindex_warning=False)
            assert len(record) == 0
        with pytest.warns(None) as record:
            evaluate_ic_list(create_ic_list_default, multiindex, multiindex_warning=False)
            assert len(record) == 0
        with pytest.warns(None) as record:
            evaluate_ic_list(multiindex, multiindex, multiindex_warning=False)
            assert len(record) == 0

    def test_perfect_match(self, create_ic_list_default):
        matches = evaluate_ic_list(create_ic_list_default, create_ic_list_default)
        assert np.all(matches["match_type"] == "tp")
        assert len(matches["match_type"] == "tp") == len(create_ic_list_default)

    def test_empty_reference(self, create_ic_list_default):
        reference = self._create_ic_list([])
        matches = evaluate_ic_list(create_ic_list_default, reference)
        assert np.all(matches["match_type"] == "fp")
        assert len(matches["match_type"] == "fp") == len(create_ic_list_default)

    def test_empty_detected(self, create_ic_list_default):
        detected = self._create_ic_list([])
        matches = evaluate_ic_list(detected, create_ic_list_default)
        assert np.all(matches["match_type"] == "fn")
        assert len(matches["match_type"] == "fn") == len(create_ic_list_default)

    def test_with_matches(self, create_ic_list_default):
        list_detected = self._create_ic_list([11, 22, 39, 45])
        matches = evaluate_ic_list(list_detected, create_ic_list_default, tolerance_samples=1)
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
        matches = evaluate_ic_list(ic_list_range_index, ic_list_range_index)
        assert np.all(matches["match_type"] == "tp")
        assert len(matches["match_type"] == "tp") == len(create_ic_list_default)
        assert_array_equal(matches.columns.to_numpy(), ["ic_id_detected", "ic_id_reference", "match_type"])
        assert matches.index.name is None
        assert_array_equal(matches.index.to_numpy(), [0, 1, 2, 3])

    def test_input_multiindex(self, create_ic_list_default):
        ic_list_multiindex = create_ic_list_default.copy()
        ic_list_multiindex.index = pd.MultiIndex.from_tuples(
            [("a", 1), ("a", 2), ("b", 3), ("b", 4)], names=["something", "ic_id"]
        )
        matches = evaluate_ic_list(ic_list_multiindex, ic_list_multiindex)
        assert np.all(matches["match_type"] == "tp")
        assert len(matches["match_type"] == "tp") == len(create_ic_list_default)
        assert_array_equal(matches.columns.to_numpy(), ["ic_id_detected", "ic_id_reference", "match_type"])
        assert matches.index.name is None
        assert_array_equal(matches.index.to_numpy(), [0, 1, 2, 3])
        assert_array_equal(ic_list_multiindex.index.to_flat_index(), matches["ic_id_detected"].to_numpy())
        assert_array_equal(ic_list_multiindex.index.to_flat_index(), matches["ic_id_reference"].to_numpy())

    def test_invalid_index_error(self, create_ic_list_default):
        ic_list_wrong_index = create_ic_list_default.copy()
        ic_list_wrong_index.index.name = "invalid"
        with pytest.raises(ValueError):
            evaluate_ic_list(ic_list_wrong_index, ic_list_wrong_index)

    def test_invalid_multiindex_error(self, create_ic_list_default):
        ic_list_wrong_index = create_ic_list_default.copy()
        ic_list_wrong_index.index = pd.MultiIndex.from_tuples(
            [("a", 1), ("a", 2), ("b", 3), ("b", 4)], names=["invalid", "also_invalid"]
        )
        with pytest.raises(ValueError):
            evaluate_ic_list(ic_list_wrong_index, ic_list_wrong_index)

    def test_segmented_stride_list_no_match(self, create_ic_list_default):
        detected = self._create_ic_list([15, 15, 35, 45])
        matches = evaluate_ic_list(detected, create_ic_list_default, tolerance_samples=4)
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

        matches = evaluate_ic_list(detected, reference, tolerance_samples=2)

        assert_array_equal(matches.query("match_type == 'tp'")["ic_id_detected"].to_numpy(), [0])
        assert_array_equal(matches.query("match_type == 'tp'")["ic_id_reference"].to_numpy(), [0])

        assert_array_equal(matches.query("match_type == 'fp'")["ic_id_detected"].to_numpy(), [1])
        assert_array_equal(matches.query("match_type == 'fp'")["ic_id_reference"].to_numpy().astype(float), [np.nan])

        assert matches.query("match_type == 'fn'").empty

        assert len(reference) == len(matches.query("match_type == 'tp'"))


class TestCalculateIcdMetrics:
    def test_ic_list_input(self, snapshot):
        ic_list_detected = pd.DataFrame([10, 20, 30, 40, 50], columns=["ic"]).rename_axis("ic_id")
        ic_list_reference = pd.DataFrame([10, 20, 40, 50, 60], columns=["ic"]).rename_axis("ic_id")
        metrics = calculate_icd_performance_metrics(
            ic_list_detected=ic_list_detected, ic_list_reference=ic_list_reference
        )
        snapshot.assert_match(pd.DataFrame(metrics, index=[0]), "metrics")

    def test_multiindex_ic_list_input(self, snapshot):
        ic_list_detected = pd.DataFrame([10, 20, 30, 40], columns=["ic"]).rename_axis("ic_id")
        ic_list_detected.index = pd.MultiIndex.from_tuples(
            [("a", 1), ("a", 2), ("b", 3), ("b", 4)], names=["something", "ic_id"]
        )
        ic_list_reference = pd.DataFrame([10, 20, 40, 50], columns=["ic"]).rename_axis("ic_id")
        ic_list_reference.index = pd.MultiIndex.from_tuples(
            [("c", 5), ("c", 6), ("d", 7), ("d", 8)], names=["something_else", "ic_id"]
        )
        # multiindex warning not suppressed per default
        with pytest.warns(Warning):
            metrics = calculate_icd_performance_metrics(
                ic_list_detected=ic_list_detected, ic_list_reference=ic_list_reference
            )
        snapshot.assert_match(pd.DataFrame(metrics, index=[0]), "metrics")
