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
        return self.ALGORITHM_CLASS(window_min=1, step_min=0.25, window_size=5, waking_hours_min=(0, 1)).detect(
            data,
            sampling_rate_hz=10.0,
        )


class TestWtdMegaritisSignal:
    def test_exactly_tiled_recording_has_no_extra_boundary_vote(self):
        data = pd.DataFrame(np.zeros((750, len(BF_SENSOR_COLS))), columns=BF_SENSOR_COLS)

        result = WtdMegaritisSignal(window_min=1, step_min=0.25, window_size=5).detect(data, sampling_rate_hz=10.0)

        assert result.diagnostics_["macro"]["start"].to_list() == [0, 150]
        assert not result.diagnostics_["macro"]["is_boundary_window"].any()
        assert result.diagnostics_["sample_votes"]["non_wear_votes"].max() == 2

    def test_complete_macro_windows_use_their_own_micro_grid(self, monkeypatch):
        data = pd.DataFrame(np.zeros((780, len(BF_SENSOR_COLS))), columns=BF_SENSOR_COLS)

        def classify_by_start(self, *, data, starts, window_samples, sampling_rate_hz):
            return starts % 40 == 0

        monkeypatch.setattr(WtdMegaritisSignal, "_classify_micro_windows_from_starts", classify_by_start)
        result = WtdMegaritisSignal(window_min=1, step_min=0.15, window_size=5, overlap=0.6).detect(
            data, sampling_rate_hz=10.0
        )

        assert result.diagnostics_["macro"]["start"].to_list() == [0, 90, 180]
        assert result.diagnostics_["macro"]["n_non_wear"].to_list() == [14, 28, 14]

    def test_custom_waking_hours_window(self):
        rng = np.random.default_rng(123)
        data = pd.DataFrame(rng.normal(size=(2400, len(BF_SENSOR_COLS))), columns=BF_SENSOR_COLS)

        result = WtdMegaritisSignal(
            window_min=1,
            step_min=0.25,
            window_size=5,
            waking_hours_min=(1, 3),
        ).detect(data, sampling_rate_hz=10.0)

        assert result.total_weartime_min_ == pytest.approx(4)
        assert result.total_weartime_during_waking_min_ == pytest.approx(2)

    def test_waking_hours_rejects_recordings_longer_than_one_day(self):
        sampling_rate_hz = 0.1
        data = pd.DataFrame(
            np.zeros((int(24 * 60 * 60 * sampling_rate_hz) + 1, len(BF_SENSOR_COLS))),
            columns=BF_SENSOR_COLS,
        )
        result = WtdMegaritisSignal(
            window_min=24 * 60 + 1,
            step_min=24 * 60 + 1,
            window_size=60,
            overlap=0.0,
            waking_hours_min=(0, 60),
        ).detect(data, sampling_rate_hz=sampling_rate_hz)

        with pytest.raises(ValueError, match="longer than one day"):
            _ = result.total_weartime_during_waking_min_

    def test_all_zero_signal_is_nonwear(self):
        data = pd.DataFrame(np.zeros((2400, len(BF_SENSOR_COLS))), columns=BF_SENSOR_COLS)

        result = WtdMegaritisSignal(window_min=1, step_min=0.25, window_size=5, waking_hours_min=(0, 1)).detect(
            data,
            sampling_rate_hz=10.0,
        )

        assert_frame_equal(result.weartime_list_, _empty_weartime_list())
        assert result.total_weartime_samples_ == 0
        assert result.total_weartime_min_ == 0

    def test_short_recording_uses_single_boundary_macro_window(self):
        data = pd.DataFrame(np.ones((700, len(BF_SENSOR_COLS))), columns=BF_SENSOR_COLS)

        result = WtdMegaritisSignal(window_min=2, step_min=0.25, window_size=5, waking_hours_min=(0, 1)).detect(
            data,
            sampling_rate_hz=10.0,
        )

        expected_macro = pd.DataFrame(
            {
                "start": [0],
                "end": [700],
                "macro_score": [1.0],
                "macro_non_wear": [True],
                "n_micro_windows": [27],
                "micro_non_wear_rate": [1.0],
                "n_wear": [0],
                "n_non_wear": [27],
                "is_boundary_window": [True],
                "is_short_recording": [True],
            }
        )

        assert_frame_equal(result.weartime_list_, _empty_weartime_list())
        assert_frame_equal(result.diagnostics_["macro"], expected_macro)

    def test_semi_simulated_wear_nonwear_regression(self):
        data = _semi_simulated_wear_nonwear_data()

        result = WtdMegaritisSignal(window_min=1, step_min=0.25, window_size=5, waking_hours_min=(0, 1)).detect(
            data,
            sampling_rate_hz=100.0,
        )

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
        assert result.total_weartime_min_ == pytest.approx(2.538)

    def test_does_not_expose_duplicate_total_weartime_units(self):
        data = pd.DataFrame(np.zeros((2400, len(BF_SENSOR_COLS))), columns=BF_SENSOR_COLS)

        result = WtdMegaritisSignal(window_min=1, step_min=0.25, window_size=5, waking_hours_min=(0, 1)).detect(
            data,
            sampling_rate_hz=10.0,
        )

        assert not hasattr(result, "total_weartime_minutes_")
        assert not hasattr(result, "total_weartime_hours_")
        assert not hasattr(result, "total_weartime_hours_during_waking_")
