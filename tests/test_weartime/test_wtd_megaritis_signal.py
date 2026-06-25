import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_ACC_COLS, BF_GYR_COLS, BF_SENSOR_COLS, GRAV_MS2
from mobgap.data import LabExampleDataset
from mobgap.utils.conversions import to_body_frame
from mobgap.weartime import WtdMegaritisSignal


def _empty_weartime_list() -> pd.DataFrame:
    return pd.DataFrame({"start": [], "end": []}).rename_axis(index="wt_id").astype("int64")


def _semi_simulated_wear_nonwear_data() -> pd.DataFrame:
    data = LabExampleDataset().get_subset(cohort="MS", participant_id="001", test="Test11", trial="Trial1").data_ss
    data = to_body_frame(data).copy()
    nonwear_start = 75 * 100
    nonwear_end = 150 * 100
    stationary_samples = nonwear_end - nonwear_start
    rng = np.random.default_rng(42)
    acc_noise = rng.normal(loc=0.0, scale=0.005, size=(stationary_samples, len(BF_ACC_COLS)))
    gyr_noise = rng.normal(loc=0.0, scale=0.005, size=(stationary_samples, len(BF_GYR_COLS)))
    stationary_acc = np.array([GRAV_MS2, 0.0, 0.0]) + acc_noise
    data.iloc[nonwear_start:nonwear_end, data.columns.get_indexer(BF_ACC_COLS)] = stationary_acc
    data.iloc[nonwear_start:nonwear_end, data.columns.get_indexer(BF_GYR_COLS)] = gyr_noise
    return data


class TestMetaWtdMegaritisSignal(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = WtdMegaritisSignal

    @pytest.fixture
    def after_action_instance(self):
        data = pd.DataFrame(np.zeros((700, len(BF_SENSOR_COLS))), columns=BF_SENSOR_COLS)
        with pytest.warns(UserWarning, match="shorter than waking hours"):
            return self.ALGORITHM_CLASS(window_min=1, step_min=0.25, window_size=5).detect(data, sampling_rate_hz=10.0)


class TestWtdMegaritisSignal:
    def test_all_zero_signal_is_nonwear(self):
        data = pd.DataFrame(np.zeros((2400, len(BF_SENSOR_COLS))), columns=BF_SENSOR_COLS)

        with pytest.warns(UserWarning, match="shorter than waking hours"):
            result = WtdMegaritisSignal(window_min=1, step_min=0.25, window_size=5).detect(data, sampling_rate_hz=10.0)

        assert_frame_equal(result.weartime_list_, _empty_weartime_list())
        assert result.total_weartime_samples_ == 0
        assert result.total_weartime_hours_ == 0

    def test_short_recording_uses_single_boundary_macro_window(self):
        data = pd.DataFrame(np.ones((400, len(BF_SENSOR_COLS))), columns=BF_SENSOR_COLS)

        with pytest.warns(UserWarning, match="shorter than waking hours"):
            result = WtdMegaritisSignal(window_min=1, step_min=0.25, window_size=5).detect(data, sampling_rate_hz=10.0)

        expected_macro = pd.DataFrame(
            {
                "start": [0],
                "end": [400],
                "macro_score": [1.0],
                "macro_non_wear": [True],
                "n_micro_windows": [15],
                "micro_non_wear_rate": [1.0],
                "n_wear": [0],
                "n_non_wear": [15],
                "is_boundary_window": [True],
                "is_short_recording": [True],
            }
        )

        assert_frame_equal(result.weartime_list_, _empty_weartime_list())
        assert_frame_equal(result.diagnostics_["macro"], expected_macro)

    def test_semi_simulated_wear_nonwear_regression(self):
        data = _semi_simulated_wear_nonwear_data()

        with pytest.warns(UserWarning, match="shorter than waking hours"):
            result = WtdMegaritisSignal(window_min=1, step_min=0.25, window_size=5).detect(data, sampling_rate_hz=100.0)

        expected_weartime = pd.DataFrame({"start": [0, 15000], "end": [7500, 22728]}).rename_axis(index="wt_id")
        expected_macro = pd.DataFrame(
            {
                "start": [0, 1500, 3000, 4500, 6000, 7500, 9000, 10500, 12000, 13500, 15000, 16500, 16728],
                "end": [6000, 7500, 9000, 10500, 12000, 13500, 15000, 16500, 18000, 19500, 21000, 22500, 22728],
                "macro_score": [0.0, 0.0, 5 / 23, 11 / 23, 17 / 23, 1.0, 1.0, 17 / 23, 11 / 23, 5 / 23, 0.0, 0.0, 0.0],
                "macro_non_wear": [
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                ],
                "n_micro_windows": [23] * 13,
                "micro_non_wear_rate": [
                    0.0,
                    0.0,
                    5 / 23,
                    11 / 23,
                    17 / 23,
                    1.0,
                    1.0,
                    17 / 23,
                    11 / 23,
                    5 / 23,
                    0.0,
                    0.0,
                    0.0,
                ],
                "n_wear": [23, 23, 18, 12, 6, 0, 0, 6, 12, 18, 23, 23, 23],
                "n_non_wear": [0, 0, 5, 11, 17, 23, 23, 17, 11, 5, 0, 0, 0],
                "is_boundary_window": [False] * 12 + [True],
                "is_short_recording": [False] * 13,
            }
        )

        assert_frame_equal(result.weartime_list_, expected_weartime)
        assert_frame_equal(result.diagnostics_["macro"], expected_macro)
        assert result.total_weartime_samples_ == 15228
        assert result.total_weartime_hours_ == pytest.approx(0.0423)
