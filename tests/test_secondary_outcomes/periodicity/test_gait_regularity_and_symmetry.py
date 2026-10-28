import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.secondary_outcomes import (
    GaitRegularityAndSymmetry,
)


class TestMetaGaitRegularityAndSymmetry(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = GaitRegularityAndSymmetry

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS().calculate(data, sampling_rate_hz=sampling_rate_hz)


class TestGaitRegularityAndSymmetry:
    def test_missing_columns_returns_self(self):
        data_missing = pd.DataFrame({"acc_is": np.random.randn(100)})
        algo = GaitRegularityAndSymmetry(replicate_matlab=False)
        result = algo.calculate(data_missing, sampling_rate_hz=100)
        assert result is algo
        assert result.signal_based_parameters_.empty
