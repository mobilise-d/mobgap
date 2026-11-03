import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.secondary_outcomes import (
    JerkRMSRatio,
)


class TestMetaJerkRMSRatio(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = JerkRMSRatio

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS().calculate(data, sampling_rate_hz=sampling_rate_hz)


class TestJerkRMSRatio:
    @pytest.mark.parametrize(
        "ml_scale,pa_scale,expected_ml,expected_pa",
        [(1.0, 1.0, 0.0, 0.0), (10.0, 0.1, 10.0, -10.0), (-10.0, -0.1, 10.0, -10.0)],
    )
    def test_slope_ratios_in_db_ignore_sign_and_unrelated_columns(self, ml_scale, pa_scale, expected_ml, expected_pa):
        time = np.arange(11) / 10
        data = pd.DataFrame(
            {"acc_is": time, "acc_ml": ml_scale * time, "acc_pa": pa_scale * time, "acc_extra": 99 * time}
        )
        row = JerkRMSRatio().calculate(data, sampling_rate_hz=10).signal_based_parameters_
        pd.testing.assert_series_equal(
            row, pd.Series({"jerk_ratio_ml_is": expected_ml, "jerk_ratio_pa_is": expected_pa})
        )

    @pytest.mark.parametrize("axis", ["ml", "pa"])
    def test_one_comparison_axis_does_not_require_the_other(self, axis):
        time = np.arange(11) / 10
        data = pd.DataFrame({"acc_is": time, f"acc_{axis}": 10 * time})
        row = JerkRMSRatio().calculate(data, sampling_rate_hz=10).signal_based_parameters_
        pd.testing.assert_series_equal(row, pd.Series({f"jerk_ratio_{axis}_is": 10.0}))

    @pytest.mark.parametrize("columns", [["acc_ml", "acc_pa"], ["acc_is"]])
    def test_missing_reference_or_comparison_axis_returns_empty(self, columns):
        time = np.arange(11) / 10
        data = pd.DataFrame(dict.fromkeys(columns, time))
        result = JerkRMSRatio().calculate(data, sampling_rate_hz=10)
        assert result.signal_based_parameters_.empty

    def test_too_few_samples_returns_empty(self):
        sampling_rate_hz = 10.0
        time = np.arange(2) / sampling_rate_hz
        data = pd.DataFrame(
            {
                "acc_is": 1.0 * time,
                "acc_ml": 1.0 * time,
            }
        )
        result = JerkRMSRatio().calculate(data, sampling_rate_hz=sampling_rate_hz)
        assert result.signal_based_parameters_.empty

    def test_zero_jerk_on_is_axis_returns_nan(self):
        sampling_rate_hz = 10.0
        time = np.arange(11) / sampling_rate_hz
        data = pd.DataFrame(
            {
                "acc_is": np.full(11, 3.0),
                "acc_ml": 1.0 * time,
            }
        )
        result = JerkRMSRatio().calculate(data, sampling_rate_hz=sampling_rate_hz)
        row = result.signal_based_parameters_
        assert np.isnan(row["jerk_ratio_ml_is"])

    def test_ratio_is_dimensionless_and_scale_invariant(self):
        sampling_rate_hz = 10.0
        time = np.arange(11) / sampling_rate_hz
        base = pd.DataFrame(
            {
                "acc_is": 1.0 * time,
                "acc_ml": 5.0 * time,
            }
        )
        scaled = pd.DataFrame(
            {
                "acc_is": 1000.0 * time,
                "acc_ml": 5000.0 * time,
            }
        )
        r1 = JerkRMSRatio().calculate(base, sampling_rate_hz=sampling_rate_hz)
        r2 = JerkRMSRatio().calculate(scaled, sampling_rate_hz=sampling_rate_hz)
        assert r1.signal_based_parameters_.loc["jerk_ratio_ml_is"] == pytest.approx(
            r2.signal_based_parameters_.loc["jerk_ratio_ml_is"]
        )
