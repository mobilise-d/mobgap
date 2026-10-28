import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.secondary_outcomes import (
    StandardDeviationAndRange,
)


class TestMetaStandardDeviationAndRange(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = StandardDeviationAndRange

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS().calculate(data)


class TestStandardDeviationAndRange:
    def test_sample_sd_range_and_units_have_known_values(self):
        row = (
            StandardDeviationAndRange()
            .calculate(pd.DataFrame({"acc_is": [0.0, 2.0, 4.0], "gyr_is": [1.0, 3.0, 5.0]}))
            .signal_based_parameters_
        )
        assert row.sd_acc_is == 2
        assert row.range_acc_is == 4
        assert row.sd_gyr_is == 2
