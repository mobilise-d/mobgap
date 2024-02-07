import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from gaitlink.utils.evaluation import (
    accuracy_score,
    f1_score,
    npv_score,
    precision_recall_f1_score,
    precision_score,
    recall_score,
    specificity_score,
)


def _create_dummy_matches_df(num_tp, num_fp, num_fn):
    tp_df = pd.DataFrame(
        np.column_stack([np.repeat(0, num_tp), np.repeat(0, num_tp), np.repeat("tp", num_tp)]),
        columns=["start", "end", "match_type"],
    )
    tp_df[["start", "end"]] = tp_df[["start", "end"]].astype(int)
    fp_df = pd.DataFrame(
        np.column_stack([np.repeat(0, num_fp), np.repeat(0, num_fp), np.repeat("fp", num_fp)]),
        columns=["start", "end", "match_type"],
    )
    fp_df[["start", "end"]] = fp_df[["start", "end"]].astype(int)
    fn_df = pd.DataFrame(
        np.column_stack([np.repeat(0, num_fn), np.repeat(0, num_fn), np.repeat("fn", num_fn)]),
        columns=["start", "end", "match_type"],
    )
    fn_df[["start", "end"]] = fn_df[["start", "end"]].astype(int)

    return pd.concat([tp_df, fp_df, fn_df])


class TestEvaluationScores:
    def test_precision(self):
        matches_df = _create_dummy_matches_df(5, 5, 5)

        precision = precision_score(matches_df)

        assert_array_equal(precision, 0.5)

    def test_perfect_precision(self):
        matches_df = _create_dummy_matches_df(5, 0, 5)

        precision = precision_score(matches_df)

        assert_array_equal(precision, 1.0)

    def test_recall(self):
        matches_df = _create_dummy_matches_df(5, 5, 5)

        recall = recall_score(matches_df)

        assert_array_equal(recall, 0.5)

    def test_perfect_recall(self):
        matches_df = _create_dummy_matches_df(5, 5, 0)

        recall = recall_score(matches_df)

        assert_array_equal(recall, 1.0)

    def test_f1_score(self):
        matches_df = _create_dummy_matches_df(5, 5, 5)

        f1 = f1_score(matches_df)

        assert_array_equal(f1, 0.5)

    def test_perfect_f1_score(self):
        matches_df = _create_dummy_matches_df(5, 0, 0)

        f1 = f1_score(matches_df)

        assert_array_equal(f1, 1.0)

    def test_precision_recall_f1(self):
        matches_df = _create_dummy_matches_df(5, 5, 5)

        eval_metrics = precision_recall_f1_score(matches_df)

        assert_array_equal(list(eval_metrics.values()), [0.5, 0.5, 0.5])

    def test_perfect_precision_recall_f1(self):
        matches_df = _create_dummy_matches_df(5, 0, 0)

        eval_metrics = precision_recall_f1_score(matches_df)

        assert_array_equal(list(eval_metrics.values()), [1.0, 1.0, 1.0])

    def test_specificity(self):
        matches_df = _create_dummy_matches_df(5, 5, 5)

        specificity = specificity_score(matches_df, 5)

        assert_array_equal(specificity, 0.5)

    def test_perfect_specificity(self):
        matches_df = _create_dummy_matches_df(0, 5, 5)

        specificity = specificity_score(matches_df, 0)

        assert_array_equal(specificity, 1.0)

    def test_accuracy(self):
        matches_df = _create_dummy_matches_df(5, 5, 5)

        accuracy = accuracy_score(matches_df, 5)

        assert_array_equal(accuracy, 0.5)

    def test_perfect_accuracy(self):
        matches_df = _create_dummy_matches_df(5, 0, 0)

        accuracy = accuracy_score(matches_df, 5)

        assert_array_equal(accuracy, 1.0)

    def test_npv_score(self):
        matches_df = _create_dummy_matches_df(5, 5, 5)

        npv = npv_score(matches_df, 5)

        assert_array_equal(npv, 0.5)

    def test_perfect_npv_score(self):
        matches_df = _create_dummy_matches_df(5, 5, 0)

        npv = npv_score(matches_df, 5)

        assert_array_equal(npv, 1.0)


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
            [specificity_score, [0, 0, 5], 0, "warn", 0],
            [specificity_score, [0, 0, 5], 0, 0, 0],
            [specificity_score, [0, 0, 5], 0, 1, 1],
            [accuracy_score, [0, 0, 0], 0, "warn", 0],
            [accuracy_score, [0, 0, 0], 0, 0, 0],
            [accuracy_score, [0, 0, 0], 0, 1, 1],
            [npv_score, [0, 5, 0], 0, "warn", 0],
            [npv_score, [0, 5, 0], 0, 0, 0],
            [npv_score, [0, 5, 0], 0, 1, 1],
        ),
    )
    def make_methods(self, request):
        self.func, self.arguments, self.tn, self.zero_division, self.expected_output = request.param

    def test_division_by_zero_return(self):
        matches_df = _create_dummy_matches_df(*self.arguments)

        if self.tn is not None:
            eval_metrics = self.func(matches_df, self.tn, zero_division=self.zero_division)
        else:
            eval_metrics = self.func(matches_df, zero_division=self.zero_division)

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
            [specificity_score, [0, 0, 5], 0, "warn", "calculating the specificity score"],
            [accuracy_score, [0, 0, 0], 0, "warn", "calculating the accuracy score"],
            [npv_score, [0, 5, 0], 0, "warn", "calculating the npv score"],
        ),
    )
    def make_methods(self, request):
        self.func, self.arguments, self.tn, self.zero_division, self.warning_message = request.param

    def test_division_by_zero_warnings(self):
        with pytest.warns(UserWarning) as w:
            if self.tn is not None:
                self.func(_create_dummy_matches_df(*self.arguments), self.tn, zero_division=self.zero_division)
            else:
                self.func(_create_dummy_matches_df(*self.arguments), zero_division=self.zero_division)

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
            [specificity_score, [0, 0, 5], 0, ""],
            [specificity_score, [0, 0, 5], 0, 2],
            [accuracy_score, [0, 0, 0], 0, ""],
            [accuracy_score, [0, 0, 0], 0, 2],
            [npv_score, [0, 5, 0], 0, ""],
            [npv_score, [0, 5, 0], 0, 2],
        ),
    )
    def make_methods(self, request):
        self.func, self.arguments, self.tn, self.zero_division = request.param

    def test_division_by_zero_warnings(self):
        with pytest.raises(ValueError) as e:
            if self.tn is not None:
                self.func(_create_dummy_matches_df(*self.arguments), self.tn, zero_division=self.zero_division)
            else:
                self.func(_create_dummy_matches_df(*self.arguments), zero_division=self.zero_division)

        # check that the message matches
        assert str(e.value) == '"zero_division" must be set to "warn", 0 or 1!'
