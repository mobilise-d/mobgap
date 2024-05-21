import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.data import LabExampleDataset
from mobgap.gsd import GsdIluz


class TestMetaGsdIluz(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = GsdIluz

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().detect(
            pd.DataFrame(np.zeros((1000, 3)), columns=["acc_x", "acc_y", "acc_z"]), sampling_rate_hz=40.0
        )


class TestGsdIluz:
    """Tests for GsdIluz.

    Note, we don't test the influence of any single parameter here.
    We don't even really know, how they all influence the results.
    We just test the happy path and some potential edegecases.
    If people run into bugs when changing parameters, we can add more tests.
    """

    def test_no_gsds(self):
        data = pd.DataFrame(np.zeros((1000, 3)), columns=["acc_x", "acc_y", "acc_z"])

        output = GsdIluz().detect(data, sampling_rate_hz=40.0).gs_list_

        assert_frame_equal(output, pd.DataFrame(columns=["start", "end"]).rename_axis("gs_id"))

    def test_single_gsd(self):
        data = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2").data_ss

        output = GsdIluz().detect(data, sampling_rate_hz=100.0).gs_list_

        assert len(output) == 1
        assert set(output.columns) == {"start", "end"}


class TestGsdIluzRegression:
    @pytest.mark.parametrize("datapoint", LabExampleDataset(reference_system="INDIP", reference_para_level="wb"))
    def test_example_lab_data(self, datapoint, snapshot):
        data = datapoint.data_ss
        sampling_rate_hz = datapoint.sampling_rate_hz

        gs_list = GsdIluz().detect(data, sampling_rate_hz=sampling_rate_hz).gs_list_
        snapshot.assert_match(gs_list, str(tuple(datapoint.group_label)))
