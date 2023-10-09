import pytest

from gaitlink.wba._wb_assembly import WBAssembly
from tests.test_wba.conftest import window


class TestWBAssembly:
    @pytest.mark.parametrize("rules", ([("test", "something_invalid")]))
    def test_invalid_rules(self, rules):
        with pytest.raises(ValueError):
            WBAssembly(rules).assemble([])

    def test_assemble_no_rules(self):
        stride_list = [window(0, 0) for _ in range(10)]

        wba = WBAssembly()
        wba.assemble(stride_list)

        assert wba.wb_list_[0]["strideList"] == stride_list
