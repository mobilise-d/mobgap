import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from gaitlink.icd import IcdShinImproved


class TestMetaShinImproved(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = IcdShinImproved

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().detect(
            pd.DataFrame(np.zeros((1000, 3)), columns=["acc_x", "acc_y", "acc_z"]), sampling_rate_hz=40.0
        )

class TestShinImprovedUnit:

    def test_invalid_axis_parameter(self):
        with pytest.raises(ValueError):
            IcdShinImproved(axis="invalid").detect(pd.DataFrame(), sampling_rate_hz=100)


class TestShinImprovedRegression:
    # TODO: Implement a no-ICs detected and an acctual regression test on example data.
    pass