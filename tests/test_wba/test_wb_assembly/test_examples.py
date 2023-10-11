from gaitlink.wba._wb_assembly import WBAssembly
from gaitlink.wba._wb_criteria import MaxBreakCriteria, NStridesCriteria
from tests.test_wba.conftest import window


def test_simple_single_wb():
    """Simple single WB test.

    Scenario:
        - Break at the beginning gait, break
        - No left right foot

    Rules:
        - BreakCriterium
        - MinStride Inclusion Rule

    """
    wb_start_time = 5
    n_strides = 10
    strides = [window(wb_start_time + i, wb_start_time + i + 1) for i in range(n_strides)]
    rules = [("break", MaxBreakCriteria(3)), ("n_strides", NStridesCriteria(4))]

    wba = WBAssembly(rules)
    wba.assemble(strides)

    assert len(wba.excluded_wb_list_) == 0
    assert len(wba.excluded_stride_list_) == 0
    assert len(wba.exclusion_reasons_) == 0
    assert len(wba.stride_exclusion_reasons_) == 0

    assert wba.wb_list_[0]["end"] == wb_start_time + n_strides
    assert wba.wb_list_[0]["start"] == wb_start_time
    assert wba.wb_list_[0]["strideList"] == strides
    assert len(wba.wb_list_) == 1


def test_simple_break_center():
    """Test gait sequence with a break in the center.

    Scenario:
        - long gait sequence with a break in the center
        - no left right

    Rules:
        - BreakCriterium
        - MinStride Inclusion Rule

    Outcome:
        - Two WBs (one for one after the break)
        - no strides discarded
    """
    wb_start_time = 5
    n_strides = 20
    strides = [window(wb_start_time + i, wb_start_time + i + 1) for i in range(n_strides)]
    del strides[7:11]
    rules = [("break", MaxBreakCriteria(3)), ("n_strides", NStridesCriteria(4))]

    wba = WBAssembly(rules)
    wba.assemble(strides)

    assert len(wba.excluded_wb_list_) == 0
    assert len(wba.excluded_stride_list_) == 0
    assert len(wba.exclusion_reasons_) == 0
    assert len(wba.stride_exclusion_reasons_) == 0

    assert len(wba.wb_list_) == 2
    assert wba.wb_list_[0]["start"] == wb_start_time
    assert wba.wb_list_[0]["end"] == wb_start_time + 7
    assert wba.wb_list_[0]["strideList"] == strides[:7]

    assert wba.wb_list_[1]["start"] == 16
    assert wba.wb_list_[1]["end"] == wb_start_time + n_strides
    assert wba.wb_list_[1]["strideList"] == strides[7:]


# TODO: Add a couple more simple test cases
