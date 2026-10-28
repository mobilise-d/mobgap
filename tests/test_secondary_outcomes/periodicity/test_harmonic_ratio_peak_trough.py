import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS
from mobgap.secondary_outcomes import (
    HarmonicRatioPeakTrough,
)


class TestMetaHarmonicRatioPeakTrough(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = HarmonicRatioPeakTrough

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS(acc_columns=["acc_is"]).calculate(
            data, stride_list=stride_list, sampling_rate_hz=sampling_rate_hz
        )


class TestHarmonicRatioPeakTrough:
    def test_empty_stride_list_returns_self(self):
        algo = HarmonicRatioPeakTrough(acc_columns=["acc_is"])
        result = algo.calculate(
            pd.DataFrame({"acc_is": np.zeros(100)}), stride_list=pd.DataFrame(), sampling_rate_hz=100
        )
        assert result is algo
        assert result.signal_based_parameters_.empty

    @pytest.mark.parametrize("offset", [0, 23])
    def test_each_supplied_stride_uses_its_exact_half_open_interval(self, offset):
        phase = np.arange(100) / 100
        stride = np.sin(4 * np.pi * phase) + 0.4 * np.sin(6 * np.pi * phase)
        data = pd.DataFrame({"acc_is": np.r_[np.full(offset, np.nan), stride, np.full(20, np.nan)]})
        strides = pd.DataFrame({"start": [offset], "end": [offset + 100]})
        row = (
            HarmonicRatioPeakTrough(acc_columns=["acc_is"])
            .calculate(data, stride_list=strides, sampling_rate_hz=100)
            .signal_based_parameters_
        )
        # Frozen coefficient-extraction result from the original audit, without endpoint adjustment.
        assert row.harmonic_ratio_acc_is == pytest.approx(25.195295004095286)

    def test_sufficient_strides_returns_float(self):
        stride_list = pd.DataFrame(
            {
                "start": [0, 60, 120],
                "end": [101, 211, 281],
                "stride_duration_s": [1.0, 1.5, 1.6],
                "stride_length_m": [0.9, 1.1, 1.0],
                "speed_mps": [0.8, 0.6, 0.7],
                "stance_time_s": [0.8, 0.9, 0.9],
                "swing_time_s": [0.2, 0.6, 0.7],
                "lr_label": ["left", "right", "left"],
                "cadence_spm": [45, 35, 50],
            }
        )
        algo = HarmonicRatioPeakTrough(acc_columns=["acc_is"])
        result = algo.calculate(
            pd.DataFrame(np.random.randn(300, 6), columns=BF_SENSOR_COLS), stride_list=stride_list, sampling_rate_hz=100
        )
        df = result.signal_based_parameters_
        assert df.shape == (1,)
        val = df.iloc[0]
        assert isinstance(val, float) and not np.isnan(val)
