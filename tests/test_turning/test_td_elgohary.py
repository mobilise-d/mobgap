import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.turning import TdElGohary


class TestMetaTdElGohary(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = TdElGohary

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().detect(
            pd.DataFrame(np.zeros((100, 3)), columns=["gyr_x", "gyr_y", "gyr_z"]), sampling_rate_hz=100.0
        )


def create_turn_series(
    centers: list[int], duration: list[int], direction: list[str], full_signal_length: int
) -> np.ndarray:
    full_signal = np.zeros(full_signal_length)
    # Each turn is a half oval with a height of 5 and a with of duration around each center
    # If the turn is a right turn, the height is negative
    for center, dur, foot in zip(centers, duration, direction):
        dur_half = dur // 2
        t = np.linspace(0, np.pi, dur)
        half_circle = 5 * np.sin(t)  # generate the y-values for the half-circle
        half_circle += 2  # shift the half-circle up by 2 to precicly control the lower threshold
        if foot == "right":
            full_signal[center - dur_half : center + dur_half] = -half_circle
        else:
            full_signal[center - dur_half : center + dur_half] = half_circle

    return full_signal


class TestTdElGohary:
    def test_no_peaks(self):
        data = pd.DataFrame(np.zeros((100, 3)), columns=["gyr_x", "gyr_y", "gyr_z"])
        output = TdElGohary()
        output.detect(data, sampling_rate_hz=100.0)
        assert output.turn_list_.empty
        assert output.raw_turn_list_.empty
        assert output.yaw_angle_.empty

    # def test_peaks_but_no_thres_crossing(self):
    #     # TODO: FIx this works for the wrong reasons!
    #     data = pd.DataFrame(np.zeros((100, 3)), columns=["gyr_x", "gyr_y", "gyr_z"])
    #     data["gyr_z"] = np.sin(np.linspace(0, 2 * np.pi, 100)) * 15 + 6 + 15
    #     output = TdElGohary()
    #     output.detect(data, sampling_rate_hz=100.0)
    #     assert output.turn_list_.empty
    #     assert output.raw_turn_list_.empty
    #     assert output.yaw_angle_.empty

    def test_sin_wave_turns(self):
        data = pd.DataFrame(np.zeros((1000, 3)), columns=["gyr_x", "gyr_y", "gyr_z"])
        # Sin wave with 10 periods = we expect 10 left and 10 right turns
        data["gyr_z"] = np.sin(np.linspace(0, 20 * np.pi, 1000))
        # We turn of the filter to avoid interference
        output = TdElGohary(smoothing_filter=None, min_peak_angle_velocity_dps=0.8, lower_threshold_velocity_dps=0.1)
        output.detect(data, sampling_rate_hz=20.0)

        assert len(output.raw_turn_list_) == 20
        turn_centers = output.raw_turn_list_["center"].to_numpy()
        # The turns should be at the peak of the sin wave
        assert np.allclose(turn_centers, np.arange(25, 1000, 50), atol=1)

    def test_individual_peaks(self):
        turns = create_turn_series([100, 200, 300], [50, 50, 50], ["left", "right", "left"], 1000)

        data = pd.DataFrame(np.zeros((1000, 3)), columns=["gyr_x", "gyr_y", "gyr_z"])
        data["gyr_z"] = turns
        output = TdElGohary(
            smoothing_filter=None,
            min_peak_angle_velocity_dps=5,
            lower_threshold_velocity_dps=2,
            allowed_turn_angle_deg=(0, np.inf),
        )

        output.detect(data, sampling_rate_hz=20.0)

        assert len(output.raw_turn_list_) == 3 == len(output.turn_list_)
        assert np.allclose(output.raw_turn_list_["center"].to_numpy(), [100, 200, 300], atol=1)
        assert np.allclose(output.turn_list_["duration_s"].to_numpy(), 50 / 20, atol=1)
        assert output.turn_list_["direction"].to_list() == ["left", "right", "left"]

    def test_merge_turns(self):
        # First two should get merged
        turns = create_turn_series([100, 155, 300, 355], [50, 50, 50, 50], ["left", "left", "right", "left"], 1000)

        data = pd.DataFrame(np.zeros((1000, 3)), columns=["gyr_x", "gyr_y", "gyr_z"])
        data["gyr_z"] = turns
        output = TdElGohary(
            smoothing_filter=None,
            min_peak_angle_velocity_dps=5,
            lower_threshold_velocity_dps=2,
            allowed_turn_angle_deg=(0, np.inf),
            min_gap_between_turns_s=10 / 20.0,
        )

        output.detect(data, sampling_rate_hz=20.0)

        assert len(output.turn_list_) == 3
        assert np.allclose(output.turn_list_["duration_s"].to_numpy(), [105 / 20, 50 / 20, 50 / 20], atol=1)
        assert output.turn_list_["direction"].to_list() == ["left", "right", "left"]
