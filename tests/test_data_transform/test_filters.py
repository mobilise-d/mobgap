import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from gaitlink.data_transform import EpflDedriftFilter, EpflGaitFilter


class TestMetaEpflGaitFilter(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = EpflGaitFilter

    @pytest.fixture()
    def after_action_instance(self):
        return EpflGaitFilter().filter(pd.DataFrame(np.zeros((500, 3))), sampling_rate_hz=40.0)


class TestMetaEpflDedriftFilter(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = EpflGaitFilter

    @pytest.fixture()
    def after_action_instance(self):
        return EpflDedriftFilter().filter(pd.DataFrame(np.zeros((500, 3))), sampling_rate_hz=40.0)
