import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.wba import MaxBreakCriteria, WbAssembly
from tests.test_wba.conftest import window


class TestWBAssemblyMeta(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = WbAssembly

    @pytest.fixture
    def after_action_instance(self, naive_stride_list) -> WbAssembly:
        return self.ALGORITHM_CLASS([("break", MaxBreakCriteria(3))]).assemble(naive_stride_list, sampling_rate_hz=1)


class TestWBAssembly:
    @pytest.mark.parametrize("rules", ([("test", "something_invalid")]))
    def test_invalid_rules(self, rules):
        with pytest.raises(ValueError):
            WbAssembly(rules).assemble(pd.DataFrame([]), sampling_rate_hz=1)

    def test_assemble_no_rules(self):
        stride_list = pd.DataFrame.from_records([window(0, 0) for _ in range(10)]).set_index("s_id")

        wba = WbAssembly()
        wba.assemble(stride_list, sampling_rate_hz=1)

        assert_frame_equal(next(iter(wba.wbs_.values())), stride_list)
        assert len(wba.wbs_) == 1
