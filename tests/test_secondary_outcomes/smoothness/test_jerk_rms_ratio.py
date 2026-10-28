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
    def test_equal_slopes_give_zero_db(self):
        sampling_rate_hz = 10.0
        time = np.arange(11) / sampling_rate_hz
        data = pd.DataFrame(
            {
                "acc_is": 1.0 * time,
                "acc_ml": 1.0 * time,
                "acc_pa": 1.0 * time,
            }
        )
        result = JerkRMSRatio().calculate(data, sampling_rate_hz=sampling_rate_hz)
        row = result.signal_based_parameters_
        assert row["jerk_ratio_ml_is"] == pytest.approx(0.0)
        assert row["jerk_ratio_pa_is"] == pytest.approx(0.0)

    def test_ml_ten_times_is_gives_10_db(self):
        sampling_rate_hz = 10.0
        time = np.arange(11) / sampling_rate_hz
        data = pd.DataFrame(
            {
                "acc_is": 1.0 * time,
                "acc_ml": 10.0 * time,
            }
        )
        result = JerkRMSRatio().calculate(data, sampling_rate_hz=sampling_rate_hz)
        row = result.signal_based_parameters_
        assert row["jerk_ratio_ml_is"] == pytest.approx(10.0)

    def test_ml_tenth_of_is_gives_minus_10_db(self):
        sampling_rate_hz = 10.0
        time = np.arange(11) / sampling_rate_hz
        data = pd.DataFrame(
            {
                "acc_is": 10.0 * time,
                "acc_ml": 1.0 * time,
            }
        )
        result = JerkRMSRatio().calculate(data, sampling_rate_hz=sampling_rate_hz)
        row = result.signal_based_parameters_
        assert row["jerk_ratio_ml_is"] == pytest.approx(-10.0)

    def test_negative_slope_uses_absolute_magnitude(self):
        sampling_rate_hz = 10.0
        time = np.arange(11) / sampling_rate_hz
        data = pd.DataFrame(
            {
                "acc_is": 1.0 * time,
                "acc_ml": -10.0 * time,
            }
        )
        result = JerkRMSRatio().calculate(data, sampling_rate_hz=sampling_rate_hz)
        row = result.signal_based_parameters_
        assert row["jerk_ratio_ml_is"] == pytest.approx(10.0)

    def test_only_ml_axis_available(self):
        sampling_rate_hz = 10.0
        time = np.arange(11) / sampling_rate_hz
        data = pd.DataFrame(
            {
                "acc_is": 1.0 * time,
                "acc_ml": 2.0 * time,
            }
        )
        result = JerkRMSRatio().calculate(data, sampling_rate_hz=sampling_rate_hz)
        row = result.signal_based_parameters_
        assert "jerk_ratio_ml_is" in result.signal_based_parameters_.index
        assert "jerk_ratio_pa_is" not in result.signal_based_parameters_.index
        assert row["jerk_ratio_ml_is"] == pytest.approx(10.0 * np.log10(2.0))

    def test_only_pa_axis_available(self):
        sampling_rate_hz = 10.0
        time = np.arange(11) / sampling_rate_hz
        data = pd.DataFrame(
            {
                "acc_is": 1.0 * time,
                "acc_pa": 2.0 * time,
            }
        )
        result = JerkRMSRatio().calculate(data, sampling_rate_hz=sampling_rate_hz)
        row = result.signal_based_parameters_
        assert "jerk_ratio_pa_is" in result.signal_based_parameters_.index
        assert "jerk_ratio_ml_is" not in result.signal_based_parameters_.index
        assert row["jerk_ratio_pa_is"] == pytest.approx(10.0 * np.log10(2.0))

    def test_missing_is_axis_returns_empty(self):
        sampling_rate_hz = 10.0
        time = np.arange(11) / sampling_rate_hz
        data = pd.DataFrame(
            {
                "acc_ml": 1.0 * time,
                "acc_pa": 1.0 * time,
            }
        )
        result = JerkRMSRatio().calculate(data, sampling_rate_hz=sampling_rate_hz)
        assert result.signal_based_parameters_.empty

    def test_single_axis_returns_empty(self):
        sampling_rate_hz = 10.0
        time = np.arange(11) / sampling_rate_hz
        data = pd.DataFrame({"acc_is": 1.0 * time})
        result = JerkRMSRatio().calculate(data, sampling_rate_hz=sampling_rate_hz)
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

    def test_handles_other_acceleration_columns_gracefully(self):
        sampling_rate_hz = 10.0
        time = np.arange(11) / sampling_rate_hz
        data = pd.DataFrame(
            {
                "acc_is": 1.0 * time,
                "acc_ml": 1.0 * time,
                "acc_extra": 99.0 * time,
            }
        )
        result = JerkRMSRatio().calculate(data, sampling_rate_hz=sampling_rate_hz)
        row = result.signal_based_parameters_
        assert set(result.signal_based_parameters_.index) == {"jerk_ratio_ml_is"}
        assert row["jerk_ratio_ml_is"] == pytest.approx(0.0)
