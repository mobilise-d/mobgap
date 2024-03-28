import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.lrd import LrdMcCamley


class TestMetaLrdMcCamley(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = LrdMcCamley

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().detect(
            pd.DataFrame(np.zeros((100, 3)), columns=["gyr_x", "gyr_y", "gyr_z"]),
            ic_list=pd.DataFrame({"ic": [5, 10, 15]}),
            sampling_rate_hz=40.0,
        )


class TestLrdMcCamley:
    def test_empty_ic(self):
        data = pd.DataFrame(np.zeros((100, 3)), columns=["gyr_x", "gyr_y", "gyr_z"])
        ic_list = pd.DataFrame({"ic": []})
        output = LrdMcCamley().detect(data, ic_list=ic_list, sampling_rate_hz=100.0).ic_lr_list_
        assert len(output) == 0
        assert list(output.columns) == ["ic", "lr_label"]

    def test_input_zero_all_left(self):
        # In the edge case of "zero" at a IC the respective label should be "left"
        data = pd.DataFrame(np.zeros((100, 3)), columns=["gyr_x", "gyr_y", "gyr_z"])
        ic_list = pd.DataFrame({"ic": [5, 10, 15]})
        output = LrdMcCamley().detect(data, ic_list=ic_list, sampling_rate_hz=100.0).ic_lr_list_
        assert len(output) == 3
        assert list(output.columns) == ["ic", "lr_label"]
        assert (output["lr_label"] == "left").all()

    @pytest.mark.parametrize("axis", ["yaw", "roll", "combined"])
    def test_simple_sin_input(self, axis):
        # A simple sine wave with a period of 10 samples
        # We expect the labels to be alternating
        data = pd.DataFrame(
            {"gyr_x": np.sin(np.linspace(0, 2 * np.pi, 100)), "gyr_z": -np.sin(np.linspace(0, 2 * np.pi, 100))}
        )
        ic_list = pd.DataFrame({"ic": [5, 15, 25]})
        output = LrdMcCamley(axis).detect(data, ic_list=ic_list, sampling_rate_hz=10.0).ic_lr_list_
        assert len(output) == 3
        assert list(output.columns) == ["ic", "lr_label"]
        assert (output["lr_label"] == ["right", "left", "right"]).all()

    @pytest.mark.parametrize("axis", ["yaw", "roll", "combined"])
    def test_correct_ouput_format(self, axis):
        data = pd.DataFrame(np.zeros((100, 3)), columns=["gyr_x", "gyr_y", "gyr_z"])
        ic_list = pd.DataFrame({"ic": [5, 10, 15]})
        output = LrdMcCamley(axis).detect(data, ic_list=ic_list, sampling_rate_hz=100.0)

        assert_frame_equal(output.ic_lr_list_[["ic"]], ic_list)

        assert isinstance(output.smoothed_data_, pd.Series)

    def test_invalid_axis(self):
        with pytest.raises(ValueError):
            LrdMcCamley(axis="invalid").detect(
                pd.DataFrame(np.zeros((100, 3)), columns=["gyr_x", "gyr_y", "gyr_z"]),
                ic_list=pd.DataFrame({"ic": [5, 10, 15]}),
                sampling_rate_hz=100.0,
            )
