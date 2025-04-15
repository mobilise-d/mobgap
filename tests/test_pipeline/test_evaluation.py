import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas._testing import assert_series_equal

from mobgap.pipeline._error_metrics import (
    abs_error,
    abs_rel_error,
    error,
    get_default_error_aggregations,
    get_default_error_transformations,
    icc,
    loa,
    quantiles,
    rel_error,
)
from mobgap.utils.df_operations import (
    CustomOperation,
    MissingDataColumnsError,
    apply_aggregations,
    apply_transformations,
)


class TestTransformationAggregationFunctions:
    def test_error_funcs(self):
        df = pd.DataFrame([[0, 1], [1, 2], [2, 3]], columns=["detected", "reference"])
        assert_series_equal(error(df), pd.Series([-1, -1, -1]))
        assert_series_equal(abs_error(df), pd.Series([1, 1, 1]))
        assert_series_equal(rel_error(df), pd.Series([-1, -1 / 2, -1 / 3]))
        assert_series_equal(abs_rel_error(df), pd.Series([1, 1 / 2, 1 / 3]))

    def test_zero_division_negative_inf(self):
        # This causes a - inf in rel error as (0 - 1) / 0 = -inf
        df = pd.DataFrame([[1, 0], [1, 2], [2, 3]], columns=["detected", "reference"])
        assert_series_equal(rel_error(df, zero_division_hint=np.nan), pd.Series([np.nan, -1 / 2, -1 / 3]))

        # Just to be sure here the +inf case
        df = pd.DataFrame([[-1, 0], [1, 2], [2, 3]], columns=["detected", "reference"])
        assert_series_equal(rel_error(df, zero_division_hint=np.nan), pd.Series([np.nan, -1 / 2, -1 / 3]))

        # And the 0 case
        df = pd.DataFrame([[0, 0], [1, 2], [2, 3]], columns=["detected", "reference"])
        assert_series_equal(rel_error(df, zero_division_hint=np.nan), pd.Series([np.nan, -1 / 2, -1 / 3]))

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
        icc_val, ci_95 = icc(df_icc, icc_type="icc1")
        assert icc_val == -1
        assert_array_equal(ci_95, [-1, -1])
        # Example for icc2
        df_perfect = pd.DataFrame({"detected": [1, 2, 3, 4], "reference": [1, 2, 3, 4]})
        icc_val, ci_95 = icc(df_perfect)
        assert round(icc_val, 3) == 1.0
        # Perfect aggrement -> variance between subjects is 0 -> CI is nan (penguin specific detail)
        assert_array_equal(np.round(ci_95, 3), [np.nan, np.nan])

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
        assert icc_val == 0
        assert_array_equal(ci_95, [np.nan, np.nan])

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

    @patch("mobgap.pipeline._error_metrics.error")
    @patch("mobgap.pipeline._error_metrics.rel_error")
    @patch("mobgap.pipeline._error_metrics.abs_error")
    @patch("mobgap.pipeline._error_metrics.abs_rel_error")
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

    @patch("mobgap.pipeline._error_metrics.error")
    @patch("mobgap.pipeline._error_metrics.rel_error")
    @patch("mobgap.pipeline._error_metrics.abs_error")
    @patch("mobgap.pipeline._error_metrics.abs_rel_error")
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
            apply_transformations(df, transformations, missing_columns="ignore")

    def test_returns_normal_col_index_instead_of_multi_index(self, combined_det_ref_dmo_df):
        # When all column names are strings and not tuples, the output should be a normal column index and not
        # Multiindex
        metric = combined_det_ref_dmo_df.columns.get_level_values(0)[0]
        transformations = [
            CustomOperation(identifier=metric, function=error, column_name="test1"),
            CustomOperation(identifier=metric, function=abs_error, column_name="test2"),
        ]
        res = apply_transformations(combined_det_ref_dmo_df, transformations)
        assert not isinstance(res.columns, pd.MultiIndex)

        # Inverted test with single value tuples
        metric = combined_det_ref_dmo_df.columns.get_level_values(0)[0]
        transformations = [
            CustomOperation(identifier=metric, function=error, column_name=("test1",)),
            CustomOperation(identifier=metric, function=abs_error, column_name=("test2",)),
        ]
        res = apply_transformations(combined_det_ref_dmo_df, transformations)
        assert isinstance(res.columns, pd.MultiIndex)


class TestApplyAggregations:
    def test_apply_agg_with_unnamed_fct(self, combined_det_ref_dmo_df):
        from functools import partial

        unnamed_trafo_func = partial(loa)
        aggs = [
            CustomOperation(
                identifier=(combined_det_ref_dmo_df.columns.get_level_values(0)[0], "reference"),
                function=unnamed_trafo_func,
                column_name=("func_name", "test_a", "test_b"),
            )
        ]
        # This used to raise an error, but #186 introduces the option to pass nameless func.
        res = apply_aggregations(combined_det_ref_dmo_df, aggs)
        assert ("func_name", "test_a", "test_b") in res.index

    @staticmethod
    def _prepare_aggregation_mock_function(mock_fct, name, return_value):
        # set names to mocked functions to prevent unnamed function error
        mock_fct.__name__ = name

        # set return values for mocked functions
        mock_fct.return_value = return_value

        return mock_fct

    def test_default_aggregations(self, combined_det_ref_dmo_df_with_errors):
        aggs = get_default_error_aggregations()
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
        assert len(res) == count_quantiles + count_icc + count_loa + count_mean + 1  # + 1 for n_datapoints
        assert res.index.get_level_values(0).nunique() == 5  # function names mean, icc, quantiles, loa
        assert res.index.get_level_values(1).nunique() == count_metrics + 1
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
        res = apply_aggregations(input_df, aggregations)

        assert isinstance(res, pd.Series)
        assert len(res) == len(functions)
        assert res.index.nlevels == (len(identifier) + 1) if isinstance(identifier, tuple) else 2

    @pytest.mark.parametrize("invalid_id", [(1,), (2, 3, 4), ["test", "test"]])
    def test_agg_aggregations_with_wrong_identifier(self, invalid_id, combined_det_ref_dmo_df_with_errors):
        functions = ["mean", "std"]
        aggregations = [(invalid_id, functions)]
        with pytest.raises(ValueError):
            apply_aggregations(combined_det_ref_dmo_df_with_errors, aggregations)

    @pytest.mark.parametrize("missing_identifier", ["bla", ("bla1", "bla2")])
    @pytest.mark.parametrize("missing_columns", ["ignore", "raise", "warn"])
    @pytest.mark.parametrize("custom_agg", [True, False])
    def test_agg_aggregations_for_missing_cols(
        self, missing_identifier, combined_det_ref_dmo_df_with_errors, missing_columns, custom_agg
    ):
        functions = ["mean", "std"]
        if not custom_agg:
            aggregations = [(missing_identifier, functions)]
        else:
            aggregations = [
                CustomOperation(identifier=missing_identifier, function=lambda x: 0, column_name=missing_identifier)
            ]

        if missing_columns in ["ignore", "warn"]:
            if missing_columns == "warn":
                with pytest.warns(UserWarning) as w:
                    res = apply_aggregations(
                        combined_det_ref_dmo_df_with_errors, aggregations, missing_columns=missing_columns
                    )
                assert len(w) == 1
            else:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    res = apply_aggregations(
                        combined_det_ref_dmo_df_with_errors, aggregations, missing_columns=missing_columns
                    )
                assert len(w) == 0
            assert_series_equal(res, pd.Series())

        elif missing_columns == "raise":
            with pytest.raises(MissingDataColumnsError):
                apply_aggregations(combined_det_ref_dmo_df_with_errors, aggregations, missing_columns=missing_columns)

        else:
            raise ValueError("Invalid value for missing_columns")

    @patch("mobgap.pipeline._error_metrics.loa")
    @patch("mobgap.pipeline._error_metrics.icc")
    @patch("mobgap.pipeline._error_metrics.quantiles")
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

    def test_split_results_to_multi_cols(self):
        def dummy_agg_func(df):
            return 1, 2

        df = pd.DataFrame(np.zeros((10, 1)), columns=["single_col"])

        aggregations = [
            CustomOperation(identifier="single_col", function=dummy_agg_func, column_name=["col_1", "col_2"])
        ]
        res = apply_aggregations(df, aggregations)
        assert_series_equal(res, pd.Series({"col_1": 1, "col_2": 2}))

    def test_split_results_to_multi_cols_invalid_length(self):
        def dummy_agg_func(df):
            return 1, 2

        df = pd.DataFrame(np.zeros((10, 1)), columns=["single_col"])

        aggregations = [CustomOperation(identifier="single_col", function=dummy_agg_func, column_name=["col_1"])]
        with pytest.raises(ValueError):
            apply_aggregations(df, aggregations)

    @patch("mobgap.pipeline._error_metrics.loa")
    @patch("mobgap.pipeline._error_metrics.quantiles")
    def test_apply_custom_aggs_with_nan_results(self, quantiles_mock, loa_mock, combined_det_ref_dmo_df_with_errors):
        quantiles_mock = self._prepare_aggregation_mock_function(quantiles_mock, "quantiles", 0)
        loa_mock = self._prepare_aggregation_mock_function(loa_mock, "loa", np.nan)

        metrics = combined_det_ref_dmo_df_with_errors.columns.get_level_values(0).unique()
        origins = combined_det_ref_dmo_df_with_errors.columns.get_level_values(1).unique()

        aggregations = [
            *[
                CustomOperation(
                    identifier=(metric, origin), function=quantiles_mock, column_name=("quantiles", metric, origin)
                )
                for metric in metrics
                for origin in origins
            ],
            *[
                CustomOperation(identifier=(metric, origin), function=loa_mock, column_name=("loa", metric, origin))
                for metric in metrics
                for origin in origins
            ],
        ]

        res = apply_aggregations(combined_det_ref_dmo_df_with_errors, aggregations)

        assert isinstance(res, pd.Series)
        assert len(res) == len(aggregations)

        assert loa_mock.call_count == len(metrics) * len(origins)
        assert quantiles_mock.call_count == len(metrics) * len(origins)

        # "quantiles", "loa"
        assert res.index.get_level_values(0).nunique() == 2
        # all metrics
        assert res.index.get_level_values(1).nunique() == len(metrics)
        # all origins
        assert res.index.get_level_values(2).nunique() == len(origins)

    def test_apply_agg_aggs_with_nan_results(self, combined_det_ref_dmo_df_with_errors):
        metrics = combined_det_ref_dmo_df_with_errors.columns.get_level_values(0).unique()

        def mock_func(x):
            return np.nan

        aggregations = [
            *(((m, o), "mean") for m in metrics for o in ["detected", "reference"]),
            *(((m, o), "std") for m in metrics for o in ["error", "rel_error"]),
            *(((m, o), mock_func) for m in metrics for o in ["detected", "abs_error"]),
        ]

        res = apply_aggregations(combined_det_ref_dmo_df_with_errors, aggregations)

        assert isinstance(res, pd.Series)
        assert len(res) == len(aggregations)

        # "mean", "std", "mock_func"
        assert res.index.get_level_values(0).nunique() == 3
        # all metrics
        assert res.index.get_level_values(1).nunique() == len(metrics)
        # "detected", "reference", "error", "rel_error", "abs_error"
        assert res.index.get_level_values(2).nunique() == 5

    @pytest.mark.parametrize(
        "incompatible_col_names", ["test", ("test", "test", "test"), ("test", "test", "test", "test")]
    )
    def test_apply_agg_with_multiple_index_level_aggregations(self, combined_det_ref_dmo_df, incompatible_col_names):
        metric = combined_det_ref_dmo_df.columns.get_level_values(0)[0]
        aggregations = [
            ((metric, "reference"), "mean"),
            CustomOperation(identifier=metric, function=lambda x: 0, column_name=incompatible_col_names),
            CustomOperation(identifier=metric, function=lambda x: 0, column_name=(metric, "test")),
        ]
        with pytest.raises(ValueError):
            apply_aggregations(combined_det_ref_dmo_df, aggregations)

    def test_apply_agg_with_empty_df(self):
        df = pd.DataFrame()
        aggs = get_default_error_aggregations()
        with pytest.raises(ValueError):
            apply_aggregations(df, aggs, missing_columns="ignore")


@pytest.fixture
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


@pytest.fixture
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
