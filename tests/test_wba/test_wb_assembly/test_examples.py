import pytest

from gaitlink.wba._wb_assembly import WBAssembly
from gaitlink.wba._wb_criteria import LevelWalkingCriteria, MaxBreakCriteria, NStridesCriteria, TurnAngleCriteria
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


@pytest.mark.parametrize("incline_strides", ([0, 1], [4, 5], [-1, -2]))
def test_incline_to_short(incline_strides):
    """Test gaitseqeunces that start with incline walking.

    Scenario:
        - 2 inline strides at the beginning/center/end
        - No left right foot

    Rules:
        - BreakCriterium
        - MinStride Inclusion Rule
        - LevelWalkingCriteria

    Outcome:
        - Single WB that includes the 2 incline strides

    """
    wb_start_time = 0
    n_strides = 10
    strides = [window(wb_start_time + i, wb_start_time + i + 1, parameter={"elevation": 0}) for i in range(n_strides)]
    for s in incline_strides:
        strides[s]["parameter"]["elevation"] = 1.0
    rules = [
        ("break", MaxBreakCriteria(3)),
        ("n_strides", NStridesCriteria(4)),
        ("level_walk", LevelWalkingCriteria(0.5, max_non_level_strides=3)),
    ]

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


@pytest.mark.parametrize("incline_strides", ([0, 1, 2, 3], [-1, -2, -3, -4]))
def test_incline_start_end_long(incline_strides):
    """Test gaitseqeunces that start with incline walking.

    Scenario:
        - 4 inline strides at the beginning (or end)
        - No left right foot

    Rules:
        - BreakCriterium
        - MinStride Inclusion Rule
        - LevelWalkingCriteria

    Outcome:
        - Single WB that starts after/ends before the incline strides
        - All incline strides are discarded
    """
    wb_start_time = 0
    n_strides = 10
    strides = [window(wb_start_time + i, wb_start_time + i + 1, parameter={"elevation": 0}) for i in range(n_strides)]
    for s in incline_strides:
        strides[s]["parameter"]["elevation"] = 1.0
    rules = [
        ("break", MaxBreakCriteria(3)),
        ("n_strides", NStridesCriteria(4)),
        ("level_walk", LevelWalkingCriteria(0.5, max_non_level_strides=3)),
    ]

    wba = WBAssembly(rules)
    wba.assemble(strides)

    assert len(wba.excluded_stride_list_) == 4
    assert len(wba.stride_exclusion_reasons_) == 4

    assert len(wba.wb_list_) == 1

    if incline_strides[-1] > 0:
        assert len(wba.excluded_wb_list_) == 0
        assert len(wba.exclusion_reasons_) == 0
        assert wba.wb_list_[0]["end"] == wb_start_time + n_strides
        assert wba.wb_list_[0]["start"] == wb_start_time + len(incline_strides)
        assert wba.wb_list_[0]["strideList"] == strides[4:]
    else:
        assert len(wba.excluded_wb_list_) == 1
        assert len(wba.exclusion_reasons_) == 1
        assert wba.wb_list_[0]["end"] == wb_start_time + n_strides - len(incline_strides)
        assert wba.wb_list_[0]["start"] == wb_start_time
        assert wba.wb_list_[0]["strideList"] == strides[:-4]


def test_incline_center_long():
    """Test gaitseqeunces that have a long incline period in the center.

    Scenario:
        - 5 inline strides in the center
        - No left right foot

    Rules:
        - BreakCriterium
        - MinStride Inclusion Rule
        - LevelWalkingCriteria

    Outcome:
        - Two WBs (one before, one after the incline period)
        - All incline strides are discarded
    """
    wb_start_time = 0
    n_strides = 15
    strides = [window(wb_start_time + i, wb_start_time + i + 1, parameter={"elevation": 0}) for i in range(n_strides)]
    incline_strides = [6, 7, 8, 9, 10]
    for s in incline_strides:
        strides[s]["parameter"]["elevation"] = 1.0
    rules = [
        ("break", MaxBreakCriteria(3)),
        ("n_strides", NStridesCriteria(4)),
        ("level_walk", LevelWalkingCriteria(0.5, max_non_level_strides=3)),
    ]

    wba = WBAssembly(rules)
    wba.assemble(strides)

    assert len(wba.excluded_wb_list_) == 0
    assert len(wba.excluded_stride_list_) == len(incline_strides)
    assert len(wba.exclusion_reasons_) == 0
    assert len(wba.stride_exclusion_reasons_) == len(incline_strides)
    for r in wba.stride_exclusion_reasons_.values():
        assert r == rules[2]

    assert len(wba.wb_list_) == 2
    assert wba.wb_list_[0]["start"] == wb_start_time
    assert wba.wb_list_[0]["end"] == wb_start_time + incline_strides[0]
    assert wba.wb_list_[0]["strideList"] == strides[: incline_strides[0]]

    assert wba.wb_list_[1]["start"] == wb_start_time + strides[incline_strides[-1] + 1]["start"]
    assert wba.wb_list_[1]["end"] == wb_start_time + n_strides
    assert wba.wb_list_[1]["strideList"] == strides[incline_strides[-1] + 1 :]


def test_short_incline_after_break():
    """Test what happens if there are incline strides after a break.

    Scenario:
        - valid WB than break, then 2 incline strides then valid wb
        - No left right

    Rules:
        - BreakCriterium
        - MinStride Inclusion Rule
        - LevelWalkingCriteria

    Outcome:
        - 2 WBs: One before the break, the second after the break
        - The second one includes the 2 incline strides
    """
    n_strides_1 = 5
    wb_start_time_1 = 0
    first_stride_list = [
        window(wb_start_time_1 + i, wb_start_time_1 + i + 1, parameter={"elevation": 0}) for i in range(n_strides_1)
    ]
    n_strides_2 = 10
    wb_start_time_2 = wb_start_time_1 + n_strides_1 + 5
    second_stride_list = [
        window(wb_start_time_2 + i, wb_start_time_2 + i + 1, parameter={"elevation": 0}) for i in range(n_strides_2)
    ]
    incline_strides = [0, 1]
    for s in incline_strides:
        second_stride_list[s]["parameter"]["elevation"] = 1.0

    strides = [*first_stride_list, *second_stride_list]
    rules = [
        ("break", MaxBreakCriteria(3)),
        ("n_strides", NStridesCriteria(4)),
        ("level_walk", LevelWalkingCriteria(0.5, max_non_level_strides=3)),
    ]

    wba = WBAssembly(rules)
    wba.assemble(strides)

    assert len(wba.excluded_wb_list_) == 0
    assert len(wba.excluded_stride_list_) == 0
    assert len(wba.exclusion_reasons_) == 0
    assert len(wba.stride_exclusion_reasons_) == 0

    assert len(wba.wb_list_) == 2

    assert wba.wb_list_[0]["start"] == wb_start_time_1
    assert wba.wb_list_[0]["end"] == wb_start_time_1 + n_strides_1
    assert wba.wb_list_[0]["strideList"] == first_stride_list

    assert wba.wb_list_[1]["start"] == wb_start_time_2
    assert wba.wb_list_[1]["end"] == wb_start_time_2 + n_strides_2
    assert wba.wb_list_[1]["strideList"] == second_stride_list


@pytest.mark.parametrize("n_incline_start", (2, 4, 1))
def test_short_incline_after_break_incline_before_break(n_incline_start):
    """Test what happens if there are incline strides before and after a break.

    Scenario:
        - some incline strides (invalid WB) than break, then 2 incline strides then valid wb
        - No left right

    Rules:
        - BreakCriterium
        - MinStride Inclusion Rule
        - LevelWalkingCriteria

    Outcome:
        - 1 WB after the break
        - The WB includes the 2 incline strides
    """
    n_strides_1 = n_incline_start
    wb_start_time_1 = 0
    first_stride_list = [
        window(wb_start_time_1 + i, wb_start_time_1 + i + 1, parameter={"elevation": 1.0}) for i in range(n_strides_1)
    ]
    n_strides_2 = 10
    wb_start_time_2 = wb_start_time_1 + n_strides_1 + 5
    second_stride_list = [
        window(wb_start_time_2 + i, wb_start_time_2 + i + 1, parameter={"elevation": 0}) for i in range(n_strides_2)
    ]
    incline_strides = [0, 1]
    for s in incline_strides:
        second_stride_list[s]["parameter"]["elevation"] = 1.0

    strides = [*first_stride_list, *second_stride_list]
    rules = [
        ("break", MaxBreakCriteria(3)),
        ("n_strides", NStridesCriteria(4)),
        ("level_walk", LevelWalkingCriteria(0.5, max_non_level_strides=3)),
    ]

    wba = WBAssembly(rules)
    wba.assemble(strides)

    assert len(wba.excluded_wb_list_) == 1
    assert len(wba.excluded_stride_list_) == n_incline_start
    assert len(wba.exclusion_reasons_) == 1
    assert len(wba.stride_exclusion_reasons_) == n_incline_start

    assert len(wba.wb_list_) == 1

    assert wba.wb_list_[0]["start"] == wb_start_time_2
    assert wba.wb_list_[0]["end"] == wb_start_time_2 + n_strides_2
    assert wba.wb_list_[0]["strideList"] == second_stride_list


@pytest.mark.parametrize("n_incline_start", (2, 4, 1))
def test_long_incline_after_break_incline_before_break(n_incline_start):
    """Test what happens if there are incline strides before and after (many) a break.

    Scenario:
        - some incline strides (invalid WB) than break, then 4 incline strides then valid wb
        - No left right

    Rules:
        - BreakCriterium
        - MinStride Inclusion Rule
        - LevelWalkingCriteria

    Outcome:
        - 1 WB after the break
        - The WB includes the 2 incline strides
    """
    n_strides_1 = n_incline_start
    wb_start_time_1 = 0
    first_stride_list = [
        window(wb_start_time_1 + i, wb_start_time_1 + i + 1, parameter={"elevation": 1.0}) for i in range(n_strides_1)
    ]
    n_strides_2 = 10
    wb_start_time_2 = wb_start_time_1 + n_strides_1 + 5
    second_stride_list = [
        window(wb_start_time_2 + i, wb_start_time_2 + i + 1, parameter={"elevation": 0}) for i in range(n_strides_2)
    ]
    incline_strides = [0, 1, 2, 3]
    for s in incline_strides:
        second_stride_list[s]["parameter"]["elevation"] = 1.0

    strides = [*first_stride_list, *second_stride_list]
    rules = [
        ("break", MaxBreakCriteria(3)),
        ("n_strides", NStridesCriteria(4)),
        ("level_walk", LevelWalkingCriteria(0.5, max_non_level_strides=3)),
    ]

    wba = WBAssembly(rules)
    wba.assemble(strides)

    assert len(wba.excluded_wb_list_) == 1
    assert len(wba.excluded_stride_list_) == n_incline_start + len(incline_strides)
    assert len(wba.exclusion_reasons_) == 1
    assert len(wba.stride_exclusion_reasons_) == n_incline_start + len(incline_strides)

    assert len(wba.wb_list_) == 1

    assert wba.wb_list_[0]["start"] == wb_start_time_2 + len(incline_strides)
    assert wba.wb_list_[0]["end"] == wb_start_time_2 + n_strides_2
    assert wba.wb_list_[0]["strideList"] == second_stride_list[len(incline_strides) :]


def test_long_incline_after_break():
    """Test what happens if there are incline strides after a break.

    Scenario:
        - valid WB than break, then 4 incline strides then valid wb
        - No left right

    Rules:
        - BreakCriterium
        - MinStride Inclusion Rule
        - LevelWalkingCriteria

    Outcome:
        - 2 WBs: One before the break, the second after the break
        - The second one should start AFTER the incline strides
    """
    n_strides_1 = 5
    wb_start_time_1 = 0
    first_stride_list = [
        window(wb_start_time_1 + i, wb_start_time_1 + i + 1, parameter={"elevation": 0}) for i in range(n_strides_1)
    ]
    n_strides_2 = 10
    wb_start_time_2 = wb_start_time_1 + n_strides_1 + 5
    second_stride_list = [
        window(wb_start_time_2 + i, wb_start_time_2 + i + 1, parameter={"elevation": 0}) for i in range(n_strides_2)
    ]
    incline_strides = [0, 1, 2, 3]
    for s in incline_strides:
        second_stride_list[s]["parameter"]["elevation"] = 1.0

    strides = [*first_stride_list, *second_stride_list]
    rules = [
        ("break", MaxBreakCriteria(3)),
        ("n_strides", NStridesCriteria(4)),
        ("level_walk", LevelWalkingCriteria(0.5, max_non_level_strides=3)),
    ]

    wba = WBAssembly(rules)
    wba.assemble(strides)

    assert len(wba.excluded_wb_list_) == 0
    assert len(wba.excluded_stride_list_) == len(incline_strides)
    assert len(wba.exclusion_reasons_) == 0
    assert len(wba.stride_exclusion_reasons_) == len(incline_strides)

    assert len(wba.wb_list_) == 2

    assert wba.wb_list_[0]["start"] == wb_start_time_1
    assert wba.wb_list_[0]["end"] == wb_start_time_1 + n_strides_1
    assert wba.wb_list_[0]["strideList"] == first_stride_list

    assert wba.wb_list_[1]["start"] == wb_start_time_2 + len(incline_strides)
    assert wba.wb_list_[1]["end"] == wb_start_time_2 + n_strides_2
    assert wba.wb_list_[1]["strideList"] == second_stride_list[4:]


def test_turns_between_wbs():
    """Test that turns correctly interrupt and restart WBs.

    Scenario:
        - long gait sequence with 2 turns in the center
        - no left/right

    Rules:
        - BreakCriterium
        - MinStride Inclusion Rule
        - TurnCriteria

    Outcome:
        - 3 WBs
        - All strides "touched" by a turn are discarded
    """
    n_strides = 20
    wb_start_time = 0
    strides = [window(wb_start_time + i, wb_start_time + i + 1) for i in range(n_strides)]
    turns = {
        "name": "turn",
        "events": [window(5, 8, parameter={"angle": 180}), window(12, 15, parameter={"angle": 180})],
    }

    rules = [
        ("break", MaxBreakCriteria(3)),
        ("n_strides", NStridesCriteria(4)),
        ("turns", TurnAngleCriteria(min_turn_angle=150, min_turn_rate=50)),
    ]

    wba = WBAssembly(rules)
    wba.assemble(strides, event_list=[turns])

    assert len(wba.excluded_wb_list_) == 0
    assert len(wba.excluded_stride_list_) == 6
    assert len(wba.exclusion_reasons_) == 0
    assert len(wba.stride_exclusion_reasons_) == 6

    for r in wba.stride_exclusion_reasons_.values():
        assert r == rules[2]

    assert len(wba.wb_list_) == 3

    assert wba.wb_list_[0]["start"] == wb_start_time
    assert wba.wb_list_[0]["end"] == 5
    assert wba.wb_list_[0]["strideList"] == strides[:5]

    assert wba.wb_list_[1]["start"] == 8
    assert wba.wb_list_[1]["end"] == 12
    assert wba.wb_list_[1]["strideList"] == strides[8:12]

    assert wba.wb_list_[2]["start"] == 15
    assert wba.wb_list_[2]["end"] == n_strides
    assert wba.wb_list_[2]["strideList"] == strides[15:]


def test_combined_turn_and_break():
    """Test what happens if a turn and a break overlap.

    Scenario:
        - A long WB with two turn and two breaks
        - The two turns and the two breaks overlap (e.g. the system was not able to detect rutning strides)
        - The first stride after the breaks have partial overlap with the turns.
    """
    n_strides = 25
    wb_start_time = 0
    strides = [window(wb_start_time + i, wb_start_time + i + 1) for i in range(n_strides)]
    del strides[13:17]
    del strides[4:7]
    turns = {
        "name": "turn",
        "events": [window(5, 7.5, parameter={"angle": 180}), window(15, 17.5, parameter={"angle": 180})],
    }

    rules = [
        ("break", MaxBreakCriteria(3)),
        ("n_strides", NStridesCriteria(4)),
        ("turns", TurnAngleCriteria(min_turn_angle=150, min_turn_rate=50)),
    ]

    wba = WBAssembly(rules)
    wba.assemble(strides, event_list=[turns])

    assert len(wba.excluded_wb_list_) == 0
    assert len(wba.excluded_stride_list_) == 2
    assert len(wba.exclusion_reasons_) == 0
    assert len(wba.stride_exclusion_reasons_) == 2

    for r in wba.stride_exclusion_reasons_.values():
        assert r == rules[2]

    assert len(wba.wb_list_) == 3

    assert wba.wb_list_[0]["start"] == wb_start_time
    assert wba.wb_list_[0]["end"] == 4
    assert wba.wb_list_[0]["strideList"] == strides[:4]

    assert wba.wb_list_[1]["start"] == 8
    assert wba.wb_list_[1]["end"] == 13
    assert wba.wb_list_[1]["strideList"] == strides[5:10]

    assert wba.wb_list_[2]["start"] == 18
    assert wba.wb_list_[2]["end"] == n_strides
    assert wba.wb_list_[2]["strideList"] == strides[11:]
