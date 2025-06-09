import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS
from mobgap.data import LabExampleDataset
from mobgap.orientation_estimation import MadgwickAHRS
from mobgap.pipeline import GsIterator
from mobgap.turning import TdElGohary
from mobgap.utils.conversions import to_body_frame


class TestMetaTdElGohary(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = TdElGohary

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().detect(
            pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS), sampling_rate_hz=100.0
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
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        output = TdElGohary()
        output.detect(data, sampling_rate_hz=100.0)
        assert output.turn_list_.empty
        assert output.raw_turn_list_.empty
        assert output.global_frame_data_ is None
        assert len(output.yaw_angle_) == len(data)

    def test_with_global_frame(self):
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        output = TdElGohary(orientation_estimation=MadgwickAHRS())
        output.detect(data, sampling_rate_hz=100.0)
        assert len(output.global_frame_data_) == len(data)

    def test_sin_wave_turns(self):
        data = pd.DataFrame(np.zeros((1000, 6)), columns=BF_SENSOR_COLS)
        # Sin wave with 10 periods = we expect 10 left and 10 right turns
        data["gyr_is"] = np.sin(np.linspace(0, 20 * np.pi, 1000))
        # We turn of the filter to avoid interference
        output = TdElGohary(smoothing_filter=None, min_peak_angle_velocity_dps=0.8, lower_threshold_velocity_dps=0.1)
        output.detect(data, sampling_rate_hz=20.0)

        assert len(output.raw_turn_list_) == 20
        turn_centers = output.raw_turn_list_["center"].to_numpy()
        # The turns should be at the peak of the sin wave
        assert np.allclose(turn_centers, np.arange(25, 1000, 50), atol=1)

    def test_individual_peaks(self):
        turns = create_turn_series([100, 200, 300], [50, 50, 50], ["left", "right", "left"], 1000)

        data = pd.DataFrame(np.zeros((1000, 6)), columns=BF_SENSOR_COLS)
        data["gyr_is"] = turns
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

        data = pd.DataFrame(np.zeros((1000, 6)), columns=BF_SENSOR_COLS)
        data["gyr_is"] = turns
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


class TestElGoharyRegression:
    @pytest.mark.parametrize("datapoint", LabExampleDataset(reference_system="INDIP", reference_para_level="wb"))
    def test_example_lab_data(self, datapoint, snapshot):
        data = to_body_frame(datapoint.data_ss)
        ref_walk_bouts = datapoint.reference_parameters_.wb_list
        if len(ref_walk_bouts) == 0:
            pytest.skip("No reference parameters available.")
        sampling_rate_hz = datapoint.sampling_rate_hz

        iterator = GsIterator()

        for (gs, data), result in iterator.iterate(data, ref_walk_bouts):
            result.turn_list = TdElGohary().detect(data, sampling_rate_hz=sampling_rate_hz).turn_list_

        detected_turns = iterator.results_.turn_list
        snapshot.assert_match(detected_turns, str(tuple(datapoint.group_label)))
