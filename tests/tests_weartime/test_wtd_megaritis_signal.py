import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS
from mobgap.data import LabExampleDataset
from mobgap.utils.conversions import to_body_frame
from mobgap.weartime import Wtd_Megaritis_signal


class TestMetaWtdMegaritisSignal(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = Wtd_Megaritis_signal

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().detect(
            pd.DataFrame(np.zeros((1000, 6)), columns=BF_SENSOR_COLS), sampling_rate_hz=100.0
        )


class TestWtdMegaritisSignal:
    """Tests for Wtd_Megaritis_signal.

    Note: We don't test the influence of any single parameter here.
    We just test the happy path and some potential edge cases.
    If people run into bugs when changing parameters, we can add more tests.
    """

    def test_no_weartime(self):
        data = pd.DataFrame(np.zeros((1000, 6)), columns=BF_SENSOR_COLS)
        output = Wtd_Megaritis_signal().detect(data, sampling_rate_hz=100.0).weartime_list_

        assert_frame_equal(output, pd.DataFrame(columns=["start", "end", "wt_id"]).astype("int64").set_index("wt_id"))

    def test_single_weartime_period(self):
        data = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2").data_ss

        output = Wtd_Megaritis_signal().detect(to_body_frame(data), sampling_rate_hz=100.0).weartime_list_

        assert len(output) >= 1
        assert set(output.columns) == {"start", "end"}


class TestWtdMegaritisSignalRegression:
    @pytest.mark.parametrize("datapoint", LabExampleDataset(reference_system="INDIP", reference_para_level="wb"))
    def test_example_lab_data(self, datapoint, snapshot):
        data = datapoint.data_ss
        sampling_rate_hz = datapoint.sampling_rate_hz

        weartime_list = (
            Wtd_Megaritis_signal().detect(to_body_frame(data), sampling_rate_hz=sampling_rate_hz).weartime_list_
        )

        snapshot.assert_match(weartime_list, str(tuple(datapoint.group_label)))
