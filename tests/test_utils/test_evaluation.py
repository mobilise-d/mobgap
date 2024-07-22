import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from mobgap.utils.evaluation import (
    accuracy_score,
    count_samples_in_intervals,
    count_samples_in_match_intervals,
    f1_score,
    npv_score,
    precision_recall_f1_score,
    precision_score,
    recall_score,
    specificity_score,
)


def _create_dummy_gsd_matches_df(num_tp, num_fp, num_fn):
    tp_df = pd.DataFrame(
        np.column_stack([np.repeat(0, num_tp), np.repeat(0, num_tp), np.repeat("tp", num_tp)]),
        columns=["start", "end", "match_type"],
    )
    tp_df[["start", "end"]] = tp_df[["start", "end"]].astype("int64")
    fp_df = pd.DataFrame(
        np.column_stack([np.repeat(0, num_fp), np.repeat(0, num_fp), np.repeat("fp", num_fp)]),
        columns=["start", "end", "match_type"],
    )
    fp_df[["start", "end"]] = fp_df[["start", "end"]].astype("int64")
    fn_df = pd.DataFrame(
        np.column_stack([np.repeat(0, num_fn), np.repeat(0, num_fn), np.repeat("fn", num_fn)]),
        columns=["start", "end", "match_type"],
    )
    fn_df[["start", "end"]] = fn_df[["start", "end"]].astype("int64")

    return pd.concat([tp_df, fp_df, fn_df])


def _create_dummy_icd_matches_df(num_tp, num_fp, num_fn):
    tp_df = pd.DataFrame(
        np.column_stack([np.repeat(0, num_tp), np.repeat(0, num_tp), np.repeat("tp", num_tp)]),
        columns=["ic_id_detected", "ic_id_reference", "match_type"],
    )
    tp_df[["ic_id_detected", "ic_id_reference"]] = tp_df[["ic_id_detected", "ic_id_reference"]].astype("int64")
    fp_df = pd.DataFrame(
        np.column_stack([np.repeat(0, num_fp), np.repeat(0, num_fp), np.repeat("fp", num_fp)]),
        columns=["ic_id_detected", "ic_id_reference", "match_type"],
    )
    fp_df[["ic_id_detected", "ic_id_reference"]] = fp_df[["ic_id_detected", "ic_id_reference"]].astype("int64")
    fn_df = pd.DataFrame(
        np.column_stack([np.repeat(0, num_fn), np.repeat(0, num_fn), np.repeat("fn", num_fn)]),
        columns=["ic_id_detected", "ic_id_reference", "match_type"],
    )
    fn_df[["ic_id_detected", "ic_id_reference"]] = fn_df[["ic_id_detected", "ic_id_reference"]].astype("int64")

    return pd.concat([tp_df, fp_df, fn_df])


class TestEvaluationScores:
    def test_precision(self):
        # test return for gsd type input
        matches_df = _create_dummy_gsd_matches_df(5, 5, 5)
        precision = precision_score(matches_df)
        assert_array_equal(precision, 0.5)

        # test return for initial_contacts type input
        matches_df = _create_dummy_icd_matches_df(5, 5, 5)
        precision = precision_score(matches_df)
        assert_array_equal(precision, 0.5)

    def test_perfect_precision(self):
        # test return for gsd type input
        matches_df = _create_dummy_gsd_matches_df(5, 0, 5)
        precision = precision_score(matches_df)
        assert_array_equal(precision, 1.0)

        # test return for initial_contacts type input
        matches_df = _create_dummy_icd_matches_df(5, 0, 5)
        precision = precision_score(matches_df)
        assert_array_equal(precision, 1.0)

    def test_recall(self):
        # test return for gsd type input
        matches_df = _create_dummy_gsd_matches_df(5, 5, 5)
        recall = recall_score(matches_df)
        assert_array_equal(recall, 0.5)

        # test return for initial_contacts type input
        matches_df = _create_dummy_icd_matches_df(5, 5, 5)
        recall = recall_score(matches_df)
        assert_array_equal(recall, 0.5)

    def test_perfect_recall(self):
        # test return for gsd type input
        matches_df = _create_dummy_gsd_matches_df(5, 5, 0)
        recall = recall_score(matches_df)
        assert_array_equal(recall, 1.0)

        # test return for initial_contacts type input
        matches_df = _create_dummy_icd_matches_df(5, 5, 0)
        recall = recall_score(matches_df)
        assert_array_equal(recall, 1.0)

    def test_f1_score(self):
        # test return for gsd type input
        matches_df = _create_dummy_gsd_matches_df(5, 5, 5)
        f1 = f1_score(matches_df)
        assert_array_equal(f1, 0.5)

        # test return for initial_contacts type input
        matches_df = _create_dummy_icd_matches_df(5, 5, 5)
        f1 = f1_score(matches_df)
        assert_array_equal(f1, 0.5)

    def test_perfect_f1_score(self):
        # test return for gsd type input
        matches_df = _create_dummy_gsd_matches_df(5, 0, 0)
        f1 = f1_score(matches_df)
        assert_array_equal(f1, 1.0)

        # test return for initial_contacts type input
        matches_df = _create_dummy_icd_matches_df(5, 0, 0)
        f1 = f1_score(matches_df)
        assert_array_equal(f1, 1.0)

    def test_precision_recall_f1(self):
        # test return for gsd type input
        matches_df = _create_dummy_gsd_matches_df(5, 5, 5)
        eval_metrics = precision_recall_f1_score(matches_df)
        assert_array_equal(list(eval_metrics.values()), [0.5, 0.5, 0.5])

        # test return for initial_contacts type input
        matches_df = _create_dummy_icd_matches_df(5, 5, 5)
        eval_metrics = precision_recall_f1_score(matches_df)
        assert_array_equal(list(eval_metrics.values()), [0.5, 0.5, 0.5])

    def test_perfect_precision_recall_f1(self):
        # test return for gsd type input
        matches_df = _create_dummy_gsd_matches_df(5, 0, 0)
        eval_metrics = precision_recall_f1_score(matches_df)
        assert_array_equal(list(eval_metrics.values()), [1.0, 1.0, 1.0])

        # test return for initial_contacts type input
        matches_df = _create_dummy_icd_matches_df(5, 0, 0)
        eval_metrics = precision_recall_f1_score(matches_df)
        assert_array_equal(list(eval_metrics.values()), [1.0, 1.0, 1.0])

    def test_specificity(self):
        # test return for gsd type input
        matches_df = _create_dummy_gsd_matches_df(5, 5, 5)
        specificity = specificity_score(matches_df, n_overall_samples=20)
        assert_array_equal(specificity, 0.5)

    def test_perfect_specificity(self):
        # test return for gsd type input
        matches_df = _create_dummy_gsd_matches_df(0, 0, 5)
        specificity = specificity_score(matches_df, n_overall_samples=10)
        assert_array_equal(specificity, 1.0)

    def test_accuracy(self):
        # test return for gsd type input
        matches_df = _create_dummy_gsd_matches_df(5, 5, 5)
        accuracy = accuracy_score(matches_df, n_overall_samples=20)
        assert_array_equal(accuracy, 0.5)

    def test_perfect_accuracy(self):
        # test return for gsd type input
        matches_df = _create_dummy_gsd_matches_df(5, 0, 0)
        accuracy = accuracy_score(matches_df, n_overall_samples=5)
        assert_array_equal(accuracy, 1.0)

    def test_npv_score(self):
        # test return for gsd type input
        matches_df = _create_dummy_gsd_matches_df(5, 5, 5)
        npv = npv_score(matches_df, n_overall_samples=20)
        assert_array_equal(npv, 0.5)

    def test_perfect_npv_score(self):
        # test return for gsd type input
        matches_df = _create_dummy_gsd_matches_df(5, 5, 0)
        npv = npv_score(matches_df, n_overall_samples=15)
        assert_array_equal(npv, 1.0)


class TestNumberSamplesLogic:
    @pytest.mark.parametrize("fct", [specificity_score, accuracy_score, npv_score])
    def test_warns_no_tn_no_n_samples(self, fct):
        matches_df = _create_dummy_gsd_matches_df(5, 5, 5)
        with pytest.warns():
            fct(matches_df)

    @pytest.mark.parametrize("fct", [specificity_score, accuracy_score, npv_score])
    def test_no_warning_when_suppressed(self, fct):
        matches_df = _create_dummy_gsd_matches_df(5, 5, 5)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            fct(matches_df, tn_warning=False)

        accuracy_score(matches_df, zero_division=0)
        npv_score(matches_df, zero_division=0)

    def test_raises_tn_and_n_samples(self):
        matches_df = _create_dummy_gsd_matches_df(5, 5, 5)
        matches_df = pd.concat(
            [
                matches_df,
                pd.DataFrame(
                    np.column_stack([np.repeat(0, 5), np.repeat(1, 5), np.repeat("tn", 5)]),
                    columns=["start", "end", "match_type"],
                ),
            ]
        )
        matches_df[["start", "end"]] = matches_df[["start", "end"]].astype("int64")

        with pytest.raises(ValueError):
            specificity_score(matches_df, n_overall_samples=20)
        with pytest.raises(ValueError):
            accuracy_score(matches_df, n_overall_samples=20)
        with pytest.raises(ValueError):
            npv_score(matches_df, n_overall_samples=20)


class TestUnsupportedInputError:
    @pytest.mark.parametrize("fct", [specificity_score, accuracy_score, npv_score])
    def test_unsupported_icd_metric_error(self, fct):
        matches_df = _create_dummy_icd_matches_df(5, 5, 5)
        with pytest.raises(ValueError):
            fct(matches_df)

    @pytest.mark.parametrize(
        "fct",
        [
            precision_score,
            recall_score,
            f1_score,
            precision_recall_f1_score,
            specificity_score,
            accuracy_score,
            npv_score,
        ],
    )
    def test_unsupported_input_error(self, fct):
        matches_df = pd.DataFrame(columns=["something", "something_else", "match_type"])
        with pytest.raises(ValueError):
            fct(matches_df)


class TestDivisionByZeroReturn:
    @pytest.fixture(
        autouse=True,
        params=(
            [precision_score, [0, 0, 5], None, "warn", 0],
            [precision_score, [0, 0, 5], None, 0, 0],
            [precision_score, [0, 0, 5], None, 1, 1],
            [recall_score, [0, 5, 0], None, "warn", 0],
            [recall_score, [0, 5, 0], None, 0, 0],
            [recall_score, [0, 5, 0], None, 1, 1],
            [f1_score, [0, 0, 0], None, "warn", 0],
            [f1_score, [0, 0, 0], None, 0, 0],
            [f1_score, [0, 0, 0], None, 1, 1],
            [precision_recall_f1_score, [0, 0, 0], None, "warn", 0],
            [precision_recall_f1_score, [0, 0, 0], None, 0, 0],
            [precision_recall_f1_score, [0, 0, 0], None, 1, 1],
            [specificity_score, [0, 0, 5], 5, "warn", 0],
            [specificity_score, [0, 0, 5], 5, 0, 0],
            [specificity_score, [0, 0, 5], 5, 1, 1],
            [accuracy_score, [0, 0, 0], 0, "warn", 0],
            [accuracy_score, [0, 0, 0], 0, 0, 0],
            [accuracy_score, [0, 0, 0], 0, 1, 1],
            [npv_score, [0, 5, 0], 5, "warn", 0],
            [npv_score, [0, 5, 0], 5, 0, 0],
            [npv_score, [0, 5, 0], 5, 1, 1],
        ),
    )
    def make_methods(self, request):
        self.func, self.arguments, self.n_samples, self.zero_division, self.expected_output = request.param

    def test_division_by_zero_return(self):
        matches_df = _create_dummy_gsd_matches_df(*self.arguments)

        # test return for gsd type input
        if self.n_samples is not None:
            eval_metrics = self.func(matches_df, n_overall_samples=self.n_samples, zero_division=self.zero_division)
        else:
            eval_metrics = self.func(matches_df, zero_division=self.zero_division)

        assert_array_equal(
            np.array(list(eval_metrics.values()) if isinstance(eval_metrics, dict) else eval_metrics),
            self.expected_output,
        )

        # test return  for initial_contacts supported metrics
        if self.n_samples is None:
            eval_metrics = self.func(_create_dummy_icd_matches_df(*self.arguments), zero_division=self.zero_division)

            assert_array_equal(
                np.array(list(eval_metrics.values()) if isinstance(eval_metrics, dict) else eval_metrics),
                self.expected_output,
            )


class TestDivisionByZeroWarnings:
    @pytest.fixture(
        autouse=True,
        params=(
            [precision_score, [0, 0, 0], None, "warn", "calculating the precision score"],
            [recall_score, [0, 5, 0], None, "warn", "calculating the recall score"],
            [f1_score, [0, 0, 0], None, "warn", "calculating the f1 score"],
            [precision_recall_f1_score, [0, 0, 0], None, "warn", "calculating the f1 score"],
            [specificity_score, [0, 0, 5], 5, "warn", "calculating the specificity score"],
            [accuracy_score, [0, 0, 0], 0, "warn", "calculating the accuracy score"],
            [npv_score, [0, 5, 0], 5, "warn", "calculating the npv score"],
        ),
    )
    def make_methods(self, request):
        self.func, self.arguments, self.n_samples, self.zero_division, self.warning_message = request.param

    def test_division_by_zero_warnings(self):
        # test return for gsd type input
        with pytest.warns(UserWarning) as w:
            if self.n_samples is not None:
                self.func(
                    _create_dummy_gsd_matches_df(*self.arguments),
                    n_overall_samples=self.n_samples,
                    zero_division=self.zero_division,
                )
            else:
                self.func(_create_dummy_gsd_matches_df(*self.arguments), zero_division=self.zero_division)

        # check that the message matches
        assert self.warning_message in w[-1].message.args[0]

        # test warning for initial_contacts supported metrics
        if self.n_samples is None:
            with pytest.warns(UserWarning) as w:
                self.func(
                    _create_dummy_icd_matches_df(*self.arguments),
                    zero_division=self.zero_division,
                )

            # check that the message matches
            assert self.warning_message in w[-1].message.args[0]


class TestDivisionByZeroError:
    @pytest.fixture(
        autouse=True,
        params=(
            [precision_score, [0, 0, 5], None, ""],
            [precision_score, [0, 0, 5], None, 2],
            [recall_score, [0, 5, 0], None, ""],
            [recall_score, [0, 5, 0], None, 2],
            [f1_score, [0, 0, 0], None, ""],
            [f1_score, [0, 0, 0], None, 2],
            [precision_recall_f1_score, [0, 0, 0], None, ""],
            [precision_recall_f1_score, [0, 0, 0], None, 2],
            [specificity_score, [0, 0, 5], 5, ""],
            [specificity_score, [0, 0, 5], 5, 2],
            [accuracy_score, [0, 0, 0], 0, ""],
            [accuracy_score, [0, 0, 0], 0, 2],
            [npv_score, [0, 5, 0], 5, ""],
            [npv_score, [0, 5, 0], 5, 2],
        ),
    )
    def make_methods(self, request):
        self.func, self.arguments, self.n_samples, self.zero_division = request.param

    def test_division_by_zero_error(self):
        # test return for gsd type input
        with pytest.raises(ValueError) as e:
            if self.n_samples is not None:
                self.func(
                    _create_dummy_gsd_matches_df(*self.arguments),
                    n_overall_samples=self.n_samples,
                    zero_division=self.zero_division,
                )
            else:
                self.func(_create_dummy_gsd_matches_df(*self.arguments), zero_division=self.zero_division)

        # check that the message matches
        assert str(e.value) == '"zero_division" must be set to "warn", 0 or 1!'

        # test error for initial_contacts supported metrics
        if self.n_samples is None:
            with pytest.raises(ValueError) as e:
                self.func(
                    _create_dummy_icd_matches_df(*self.arguments),
                    zero_division=self.zero_division,
                )

            # check that the message matches
            assert str(e.value) == '"zero_division" must be set to "warn", 0 or 1!'


class TestCountSamples:
    def test_count_samples_in_intervals(self):
        intervals = pd.DataFrame({"start": [0, 2], "end": [1, 4]})
        count = count_samples_in_intervals(intervals)

        assert count == 5

    def test_count_samples_in_intervals_empty(self):
        intervals = pd.DataFrame(columns=["start", "end", "match_type"])
        count = count_samples_in_match_intervals(intervals, "tp")

        assert count == 0

    def test_count_samples_in_intervals_invalid_input(self):
        intervals = pd.DataFrame(columns=["start", "not_end"])
        with pytest.raises(ValueError):
            count_samples_in_intervals(intervals)

    def test_count_samples_in_match_intervals(self):
        intervals = pd.DataFrame(
            {"start": [0, 1, 2, 3, 4], "end": [1, 2, 3, 4, 5], "match_type": ["tp", "fp", "fn", "tn", "tp"]}
        )
        tp_count = count_samples_in_match_intervals(intervals, "tp")
        fp_count = count_samples_in_match_intervals(intervals, "fp")
        fn_count = count_samples_in_match_intervals(intervals, "fn")
        tn_count = count_samples_in_match_intervals(intervals, "tn")

        assert tp_count == 4
        assert fp_count == 2
        assert fn_count == 2
        assert tn_count == 2

    def test_count_samples_in_match_intervals_empty(self):
        intervals = pd.DataFrame(columns=["start", "end", "match_type"])
        tp_count = count_samples_in_match_intervals(intervals, "tp")
        fp_count = count_samples_in_match_intervals(intervals, "fp")
        fn_count = count_samples_in_match_intervals(intervals, "fn")
        tn_count = count_samples_in_match_intervals(intervals, "tn")

        assert tp_count == 0
        assert fp_count == 0
        assert fn_count == 0
        assert tn_count == 0

    def test_count_samples_in_match_intervals_invalid_input(self):
        intervals = pd.DataFrame(columns=["start", "end", "not_match_type"])
        with pytest.raises(ValueError):
            count_samples_in_match_intervals(intervals, "tp")
