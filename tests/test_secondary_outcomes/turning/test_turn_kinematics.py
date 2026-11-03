import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.secondary_outcomes import (
    TurnKinematics,
)


class TestMetaTurnKinematics(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = TurnKinematics

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        turn = turn_list.iloc[0]
        return self.ALGORITHM_CLASS().calculate(
            data.iloc[int(turn.start) : int(turn.end)], sampling_rate_hz=sampling_rate_hz
        )


class TestTurnKinematics:
    def test_empty_window_returns_empty_series(self):
        result = TurnKinematics().calculate(pd.DataFrame({"gyr_is": []}), sampling_rate_hz=100)
        pd.testing.assert_series_equal(result.signal_based_parameters_, pd.Series(dtype=float))

    @pytest.mark.parametrize("yaw,mean,peak", [([-3.0, -1.0, 1.0, 3.0], 0.0, 3.0), ([2.0, 2.0, 6.0, 6.0], 4.0, 6.0)])
    def test_one_window_retains_all_turn_values_and_signed_mean(self, yaw, mean, peak):
        row = TurnKinematics().calculate(pd.DataFrame({"gyr_is": yaw}), sampling_rate_hz=10).signal_based_parameters_
        assert row["turn_mean_ang_vel"] == mean
        assert row["turn_peak_ang_vel"] == peak
        assert "turn_smoothness" in row.index
        if mean == 0:
            assert np.isnan(row["turn_smoothness"])

    @pytest.mark.parametrize("yaw,mean,peak", [([2.0], 2.0, 2.0), ([2.0, 4.0], 3.0, 4.0), ([2.0] * 4, 2.0, 2.0)])
    def test_short_or_constant_turn_retains_velocities_with_undefined_smoothness(self, yaw, mean, peak):
        row = TurnKinematics().calculate(pd.DataFrame({"gyr_is": yaw}), sampling_rate_hz=10).signal_based_parameters_
        assert row.turn_mean_ang_vel == mean
        assert row.turn_peak_ang_vel == peak
        assert np.isnan(row.turn_smoothness)

    def test_turn_smoothness_is_angular_ldlj(self):
        data = pd.DataFrame({"gyr_is": [999.0, *((np.arange(11) / 10) ** 2), 999.0]})
        row = TurnKinematics().calculate(data.iloc[1:12], sampling_rate_hz=10).signal_based_parameters_
        assert row.turn_smoothness == pytest.approx(-np.log(3.35 / 0.65**2))
