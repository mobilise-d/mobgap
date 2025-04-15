import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.walking_speed import WsNaive


class TestMetaWsNaive(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = WsNaive

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().calculate(
            data=pd.DataFrame(np.zeros((100, 3)), columns=["acc_x", "acc_y", "acc_z"]),
            initial_contacts=None,
            cadence_per_sec=pd.DataFrame({"cadence_spm": np.arange(0, 5, 1)}),
            stride_length_per_sec=pd.DataFrame({"stride_length_m": np.arange(0, 5, 1)}),
            sampling_rate_hz=100.0,
        )


class TestWsNaive:
    def test_simple(self):
        i = pd.Index(np.arange(0, 5, 1), name="bla")
        cadence = pd.DataFrame({"cadence_spm": [120, 130, 140, 150, 160]}, index=i)
        stride_length = pd.DataFrame({"stride_length_m": [1.2, 1.3, 1.4, 1.5, 1.6]}, index=i)
        # Data can be empty
        data = pd.DataFrame([])
        # Can also be empty
        sampling_rate_hz = 1

        ws_naive = WsNaive().calculate(
            data=data, cadence_per_sec=cadence, stride_length_per_sec=stride_length, sampling_rate_hz=sampling_rate_hz
        )

        assert_frame_equal(
            ws_naive.walking_speed_per_sec_,
            pd.DataFrame(
                {"walking_speed_mps": (cadence.to_numpy() * stride_length.to_numpy() / (60 * 2)).flatten()}, index=i
            ),
        )

    @pytest.mark.parametrize("para", ["cadence_per_sec", "stride_length_per_sec"])
    def test_requires_cadence_and_sl(self, para):
        i = pd.Index(np.arange(0, 5, 1), name="bla")
        cadence = pd.DataFrame({"cadence_spm": [120, 130, 140, 150, 160]}, index=i)
        stride_length = pd.DataFrame({"stride_length_m": [1.2, 1.3, 1.4, 1.5, 1.6]}, index=i)
        # Data can be empty
        data = pd.DataFrame([])
        # Can also be empty
        sampling_rate_hz = 1

        if para == "cadence_per_sec":
            cadence = None
        else:
            stride_length = None

        with pytest.raises(ValueError):
            WsNaive().calculate(
                data=data,
                cadence_per_sec=cadence,
                stride_length_per_sec=stride_length,
                sampling_rate_hz=sampling_rate_hz,
            )
