import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.secondary_outcomes import (
    JerkRMSRatio,
    LogDimensionlessJerk,
)


class TestMetaLogDimensionlessJerk(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = LogDimensionlessJerk

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS(
            acc_columns=["acc_is", "acc_ml", "acc_pa"], gyr_columns=["gyr_is", "gyr_ml", "gyr_pa"]
        ).calculate(data, sampling_rate_hz=sampling_rate_hz)


class TestLogDimensionlessJerk:
    def test_linear_acceleration_has_constant_jerk(self):
        sampling_rate_hz = 10.0
        time = np.arange(11) / sampling_rate_hz
        data = pd.DataFrame({"acc_is": 2.0 * time})
        result = LogDimensionlessJerk(acc_columns=["acc_is"]).calculate(data, sampling_rate_hz=sampling_rate_hz)
        assert result.signal_based_parameters_.loc["jerk_acc_is"] == pytest.approx(-np.log(4))

    def test_constant_acceleration_gives_nan_ldlj(self):
        sampling_rate_hz = 10.0
        data = pd.DataFrame({"acc_is": np.full(11, 3.0)})
        result = LogDimensionlessJerk(acc_columns=["acc_is"]).calculate(data, sampling_rate_hz=sampling_rate_hz)
        assert np.isnan(result.signal_based_parameters_.loc["jerk_acc_is"])

    def test_quadratic_velocity_has_finite_ldlj_v(self):
        sampling_rate_hz = 10.0
        time = np.arange(11) / sampling_rate_hz
        data = pd.DataFrame({"gyr_is": time**2})
        result = LogDimensionlessJerk(gyr_columns=["gyr_is"]).calculate(data, sampling_rate_hz=sampling_rate_hz)
        assert result.signal_based_parameters_.loc["jerk_gyr_is"] == pytest.approx(-np.log(3.35 / 0.65**2))

    def test_linear_velocity_gives_nan_ldlj_v(self):
        sampling_rate_hz = 10.0
        time = np.arange(11) / sampling_rate_hz
        data = pd.DataFrame({"gyr_is": 2.0 * time})
        result = LogDimensionlessJerk(gyr_columns=["gyr_is"]).calculate(data, sampling_rate_hz=sampling_rate_hz)
        assert np.isnan(result.signal_based_parameters_.loc["jerk_gyr_is"])

    def test_missing_columns_are_silently_skipped(self):
        algo = LogDimensionlessJerk(
            acc_columns=["acc_is", "acc_missing"],
            gyr_columns=["gyr_missing"],
        )
        data = pd.DataFrame(
            np.random.default_rng(2).standard_normal((300, 1)),
            columns=["acc_is"],
        )
        result = algo.calculate(data, sampling_rate_hz=100.0)
        assert set(result.signal_based_parameters_.index) == {"jerk_acc_is"}

    def test_no_columns_returns_empty(self):
        algo = LogDimensionlessJerk()
        data = pd.DataFrame(
            np.random.default_rng(3).standard_normal((300, 3)),
            columns=["acc_is", "acc_ml", "acc_pa"],
        )
        result = algo.calculate(data, sampling_rate_hz=100.0)
        assert result.signal_based_parameters_.empty

    def test_ldlj_is_scale_invariant(self):
        sampling_rate_hz = 10.0
        time = np.arange(11) / sampling_rate_hz
        base = pd.DataFrame({"acc_is": 2.0 * time, "gyr_is": time**2})
        scaled = pd.DataFrame({"acc_is": 200.0 * time, "gyr_is": 100.0 * time**2})
        algo = LogDimensionlessJerk(acc_columns=["acc_is"], gyr_columns=["gyr_is"])
        r1 = algo.calculate(base, sampling_rate_hz=sampling_rate_hz).signal_based_parameters_
        r2 = algo.calculate(scaled, sampling_rate_hz=sampling_rate_hz).signal_based_parameters_
        assert r1["jerk_acc_is"] == pytest.approx(r2["jerk_acc_is"])
        assert r1["jerk_gyr_is"] == pytest.approx(r2["jerk_gyr_is"])

    def test_ldlj_is_shift_invariant(self):
        sampling_rate_hz = 10.0
        time = np.arange(11) / sampling_rate_hz
        base = pd.DataFrame({"acc_is": 2.0 * time, "gyr_is": time**2})
        shifted = pd.DataFrame({"acc_is": 2.0 * time + 9.81, "gyr_is": time**2 + 5.0})
        algo = LogDimensionlessJerk(acc_columns=["acc_is"], gyr_columns=["gyr_is"])
        r1 = algo.calculate(base, sampling_rate_hz=sampling_rate_hz).signal_based_parameters_
        r2 = algo.calculate(shifted, sampling_rate_hz=sampling_rate_hz).signal_based_parameters_
        assert r1["jerk_acc_is"] == pytest.approx(r2["jerk_acc_is"])
        assert r1["jerk_gyr_is"] == pytest.approx(r2["jerk_gyr_is"])

    def test_too_few_samples_returns_nan(self):
        algo = LogDimensionlessJerk(acc_columns=["acc_is"])
        data = pd.DataFrame({"acc_is": [1.0, 2.0]})
        result = algo.calculate(data, sampling_rate_hz=10.0)
        assert np.isnan(result.signal_based_parameters_.loc["jerk_acc_is"])


@pytest.mark.parametrize(
    "duration,acc_expected,gyr_expected",
    [(5, -6.20017, -13.09348), (10, -7.58646, -15.86607), (20, -8.97276, -18.63866)],
)
def test_repeated_motion_changes_ldlj_but_preserves_rms_jerk_ratio(duration, acc_expected, gyr_expected):
    t = np.arange(duration * 100 + 1) / 100
    sine = np.sin(2 * np.pi * t)
    data = pd.DataFrame({"acc_is": sine, "acc_ml": 3 * sine, "gyr_is": sine})
    row = (
        LogDimensionlessJerk(acc_columns=["acc_is"], gyr_columns=["gyr_is"])
        .calculate(data, sampling_rate_hz=100)
        .signal_based_parameters_
    )
    assert row["jerk_acc_is"] == pytest.approx(acc_expected, abs=1e-5)
    assert row["jerk_gyr_is"] == pytest.approx(gyr_expected, abs=1e-5)
    ratio = JerkRMSRatio().calculate(data, sampling_rate_hz=100).signal_based_parameters_
    assert ratio["jerk_ratio_ml_is"] == pytest.approx(4.771212547)
