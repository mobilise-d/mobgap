import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.turning import TdElGohary


class TestMetaHKLeeImproved(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = TdElGohary

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().detect(
            pd.DataFrame(np.zeros((1000, 3)), columns=["gyr_x", "gyr_y", "gyr_z"]), sampling_rate_hz=120.0
        )
