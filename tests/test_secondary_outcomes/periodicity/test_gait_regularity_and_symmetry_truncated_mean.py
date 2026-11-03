import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.secondary_outcomes import GaitRegularityAndSymmetryTruncatedMean


class TestMetaGaitRegularityAndSymmetryTruncatedMean(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = GaitRegularityAndSymmetryTruncatedMean

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS().calculate(data, sampling_rate_hz=sampling_rate_hz)
