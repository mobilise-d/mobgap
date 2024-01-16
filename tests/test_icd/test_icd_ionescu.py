import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from tpcp.testing import TestAlgorithmMixin

from gaitlink.data import LabExampleDataset
from gaitlink.icd import IcdIonescu


class TestMetaIcdIonescu(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = IcdIonescu

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().detect(
            pd.DataFrame(np.zeros((1000, 3)), columns=["acc_x", "acc_y", "acc_z"]), sampling_rate_hz=40.0
        )


class TestIcdIonescu:
    """Tests for IcdIonescu.

    We just test the happy path and some potential edgecases.
    If people run into bugs when changing parameters, we can add more tests.
    """

    def test_no_icds(self):
        data = pd.DataFrame(np.zeros((1000, 3)), columns=["acc_x", "acc_y", "acc_z"])  # not a gait sequence

        output = IcdIonescu().detect(data, sampling_rate_hz=100.0).icd_list_

        assert_series_equal(output, pd.Series(name="ic").astype(float))

    def test_single_icd(self):
        # s and e delimit a gait sequence with just one IC
        s = 700
        e = 800
        data = (
            LabExampleDataset()
            .get_subset(cohort="MS", participant_id="001", test="Test5", trial="Trial1")
            .data["LowerBack"][s : e + 1]
        )

        output = IcdIonescu().detect(data, sampling_rate_hz=100.0).icd_list_

        assert len(output) == 1
        assert set(output.name) == "ic"
