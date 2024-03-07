import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from gaitlink.cad import CadFromIc


class TestMetaCadFromIc(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = CadFromIc

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().calculate(
            pd.DataFrame(np.zeros((100, 3)), columns=["acc_x", "acc_y", "acc_z"]),
            initial_contacts=pd.Series(np.arange(0, 100, 5)),
            sampling_rate_hz=40.0,
        )
