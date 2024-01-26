import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from gaitlink.data import LabExampleDataset
from gaitlink.gsd import GsdIluz


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

        assert_frame_equal(output, pd.DataFrame(columns=["start", "end"]))

    def test_single_gsd(self):
        data = (
            LabExampleDataset()
            .get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2")
            .data["LowerBack"]
        )

        output = GsdIluz().detect(data, sampling_rate_hz=100.0).gs_list_

        assert len(output) == 1
        assert set(output.columns) == {"start", "end"}
