import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.sl import SlZijlstra
from mobgap.data import LabExampleDataset
from mobgap.sl.base import BaseSlCalculator
from mobgap.pipeline import GsIterator


class TestMetaSlZijlstra(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = SlZijlstra

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().calculate(
            pd.DataFrame(np.zeros((100, 3)), columns=["acc_x", "acc_y", "acc_z"]),
            initial_contacts=pd.DataFrame({"ic": np.arange(0, 100, 5)}),
            sampling_rate_hz=40.0,
        )

