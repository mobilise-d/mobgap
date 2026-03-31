import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS
from mobgap.laterality import LrcMansour


class TestMetaLrcMansour(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = LrcMansour

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().predict(
            pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS),
            ic_list=pd.DataFrame({"ic": [5, 10, 15]}),
            sampling_rate_hz=40.0,
        )


class TestLrcMansour:
    def test_empty_ic(self):
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        ic_list = pd.DataFrame({"ic": []})
        output = LrcMansour().predict(data, ic_list=ic_list, sampling_rate_hz=100.0).ic_lr_list_
        assert len(output) == 0
        assert list(output.columns) == ["ic", "lr_label"]

    def test_input_zero_all_right(self):
        # In the edge case of "zero" at a IC the respective label should be "right"
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        ic_list = pd.DataFrame({"ic": [5, 10, 15]})
        output = LrcMansour().predict(data, ic_list=ic_list, sampling_rate_hz=100.0).ic_lr_list_
        assert len(output) == 3
        assert list(output.columns) == ["ic", "lr_label"]
        assert (output["lr_label"] == "right").all()

    def test_simple_sin_input(self):
        # A simple sine wave with two full periods
        # We expect the labels to be alternating
        # when looking at points at falling, rising, falling
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        data["acc_ml"] = np.sin(np.linspace(0, 4 * np.pi, 100))

        ic_list = pd.DataFrame({"ic": [20, 50, 70]})
        output = LrcMansour().predict(data, ic_list=ic_list, sampling_rate_hz=10.0).ic_lr_list_
        assert len(output) == 3
        assert list(output.columns) == ["ic", "lr_label"]
        assert (output["lr_label"] == ["right", "left", "right"]).all()

    def test_correct_ouput_format(self):
        data = pd.DataFrame(np.zeros((100, 6)), columns=BF_SENSOR_COLS)
        ic_list = pd.DataFrame({"ic": [5, 10, 15]})
        output = LrcMansour().predict(data, ic_list=ic_list, sampling_rate_hz=100.0)

        assert_frame_equal(output.ic_lr_list_[["ic"]], ic_list)

        assert isinstance(output.smoothed_data_, pd.Series)
