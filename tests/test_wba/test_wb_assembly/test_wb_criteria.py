from itertools import repeat

import numpy as np
import pytest

from gaitlink.wba._wb_assembly import WBAssembly, create_wb
from gaitlink.wba._wb_criteria import (
    LeftRightCriteria,
    LevelWalkingCriteria,
    MaxBreakCriteria,
    NStridesCriteria,
    TurnAngleCriteria,
)
from gaitlink.wba._wb_criteria_base import EventTerminationCriteria, WBCriteria
from tests.test_wba.conftest import window


class BaseTestCriteriaInclusion:
    criteria_class: type[WBCriteria]

    defaults = {}

    def test_single_stride(self, naive_stride_list, naive_event_list):
        """Test that now exception is thrown if a list with a single stride is checked."""
        single_stride_list = [naive_stride_list[0]]

        self.criteria_class(**self.defaults).check_include(create_wb(single_stride_list), naive_event_list)


class BaseTestCriteriaTermination:
    criteria_class: type[WBCriteria]

    defaults = {}

    def test_single_stride(self, naive_stride_list, naive_event_list):
        """Test that no exception is thrown if a list with a single stride is checked."""
        single_stride_list = [naive_stride_list[0]]

        self.criteria_class(**self.defaults).check_wb_start_end(single_stride_list, 0, 0, 0, naive_event_list)


class TestMaxBreakCriteria(BaseTestCriteriaTermination):
    criteria_class = MaxBreakCriteria
    defaults = {"max_break": 0, "remove_last_ic": False, "comment": "comment"}

    @pytest.fixture(
        params=(
            (
                {"max_break": 0, "comment": "comment"},
                {"max_break": 0, "comment": "comment"},
            ),
            ({"max_break": -5, "comment": None}, ValueError),
            ({"max_break": -5, "remove_last_ic": "something_wrong", "comment": None}, ValueError),
            (
                {"max_break": 100, "remove_last_ic": True, "name": None, "comment": None},
                {"max_break": 100, "remove_last_ic": True, "name": None, "comment": None},
            ),
            (
                {"max_break": 100, "remove_last_ic": "per_foot", "name": None, "comment": None},
                {"max_break": 100, "remove_last_ic": "per_foot", "name": None, "comment": None},
            ),
        )
    )
    def init_data(self, request):
        return request.param

    @pytest.mark.parametrize(
        ("new_stride_break", "allowed_break", "expected"),
        ((0, 2, True), (2, 2, True), (3, 2, False)),
    )
    def test_check_simple(self, naive_stride_list, new_stride_break, allowed_break, expected):
        new_stride_start = naive_stride_list[-1]["end"] + new_stride_break
        new_stride = window(start=new_stride_start, end=new_stride_start + 100)
        naive_stride_list.append(new_stride)
        c = MaxBreakCriteria(allowed_break)

        expected = None if expected is True else len(naive_stride_list) - 2

        assert (
            c.check_wb_start_end(
                stride_list=naive_stride_list, original_start=0, current_start=0, current_end=len(naive_stride_list) - 1
            )[1]
            is expected
        )


#     TODO: Test new delay rules


class TestNStridesCriteria(BaseTestCriteriaInclusion):
    criteria_class = NStridesCriteria
    defaults = {"min_strides": 1, "comment": "comment"}

    @pytest.fixture(
        params=(
            (
                {
                    "min_strides": 1,
                    "min_strides_left": 3,
                    "min_strides_right": 4,
                    "comment": "comment",
                },
                {
                    "min_strides": 1,
                    "min_strides_left": 3,
                    "min_strides_right": 4,
                    "comment": "comment",
                },
            ),
        )
    )
    def init_data(self, request):
        return request.param

    @pytest.mark.parametrize(
        ("lower", "length", "result"),
        (
            (0, 1, True),
            (0, 2, True),
            (1, 1, True),
            (5, 2, False),
            (None, 1, False),
        ),
    )
    def test_check_no_foot(self, lower, length, result):
        c = NStridesCriteria(lower)
        stride_list = [window(0, 0) for _ in range(length)]
        wb = create_wb(stride_list)
        assert c.check_include(wb) == result

    @pytest.mark.parametrize(
        ("lower", "lower_left", "lower_right", "nleft", "nright", "nrest", "result"),
        (
            (None, 0, 0, 1, 0, 0, True),
            (None, 3, 3, 4, 0, 0, True),
            (None, 3, 3, 3, 0, 0, True),
            (4, 3, 3, 3, 3, 5, True),
            (None, 3, 3, 0, 0, 5, False),
            (4, 3, 3, 0, 0, 5, True),
        ),
    )
    def test_check_with_foot(self, lower, lower_left, lower_right, nleft, nright, nrest, result):
        c = NStridesCriteria(lower, min_strides_left=lower_left, min_strides_right=lower_right)
        left = [window(0, 0, foot="left") for _ in range(nleft)]
        right = [window(0, 0, foot="right") for _ in range(nright)]
        rest = [window(0, 0, foot=None) for _ in range(nrest)]
        stride_list = [*left, *right, *rest]
        wb = create_wb(stride_list)
        assert c.check_include(wb) == result


class TestLeftRightCriteria(BaseTestCriteriaTermination):
    criteria_class = LeftRightCriteria
    defaults = {"comment": "comment"}

    @pytest.fixture(
        params=(
            (
                {"comment": "comment"},
                {"comment": "comment"},
            ),
        )
    )
    def init_data(self, request):
        return request.param

    @pytest.mark.parametrize(
        ("feet", "result"),
        (
            (["left", "left"], 0),
            (["right", "right"], 0),
            (["right", "left"], None),
            (["left", "right"], None),
        ),
    )
    def test_check(self, feet, result):
        c = LeftRightCriteria()
        stride_list = [window(0, 0, foot=f) for f in feet]
        out = c.check_wb_start_end(stride_list, current_start=0, original_start=0, current_end=len(stride_list) - 1)
        assert out[1] == result
        assert out[0] is None


class TestEventTerminationCriteria(BaseTestCriteriaTermination):
    criteria_class = EventTerminationCriteria
    defaults = {"event_type": "event1", "termination_mode": "ongoing", "comment": "comment"}

    test_cases = (
        (((0, 1), (1.5, 2.5)), (1.2, 1.3)),  # 1 Started and stopped between strides
        (
            ((0, 1), (1.5, 2.5)),
            (0.5, 2),
        ),  # 2 Started in last stride and stopped in current strides
        (
            ((0, 1), (1.5, 2.5)),
            (1, 1.2),
        ),  # 3 Started at end of last stride and stopped between strides
        (((0, 1), (1.5, 2.5)), (1.7, 3)),  # 4 Started in current stride and ended after
        (((0, 1), (1.5, 2.5)), (1.7, 2)),  # 5 Started and ended in current stride
        (((0, 1), (1.5, 2.5)), (0.2, 0.8)),  # 6 Started and ended in last stride
        (
            ((1, 2), (2.5, 3.5)),
            (0.5, 4.5),
        ),  # 7 Started before last stride and stopped after current strides
        # Cases that are not checked because they occur before the last stride
        (((1, 2), (2.5, 3.5)), (0.5, 0.8)),  # 8 Started and stopped before last stride
        # Cases that are not checked because they occur after the current stride
        (((0, 1), (1.5, 2.5)), (3, 3.5)),  # 9 Started and stopped after current stride
        # Edge cases
        (
            ((0, 1), (1.5, 2.5)),
            (1, 2),
        ),  # 10 Started at end of last stride and stopped in current strides
        (
            ((0, 1), (1.5, 2.5)),
            (1, 1.2),
        ),  # 11 Started at end of last stride and stopped between strides
        (((0, 1), (1.5, 2.5)), (0, 1)),  # 12 exactly the last stride
        (((0, 1), (1.5, 2.5)), (1.5, 2.5)),  # 13 exactly the current stride
        (((0, 1), (1.5, 2.5)), (0, 2.5)),  # 14 exactly the both strides
        (((0, 1), (1.5, 2.5)), (2.5, 3)),  # 15 start at end of last stride
    )

    start_results = (
        False,  # 1
        False,  # 2
        False,  # 3
        False,  # 4
        False,  # 5
        False,  # 6
        True,  # 7
        True,  # 8
        True,  # 9
        False,  # 10
        False,  # 11
        False,  # 12
        False,  # 13
        False,  # 14
        True,  # 15
    )

    end_results = (
        False,  # 1
        False,  # 2
        False,  # 3
        True,  # 4
        False,  # 5
        False,  # 6
        True,  # 7
        True,  # 8
        True,  # 9
        False,  # 10
        False,  # 11
        False,  # 12
        True,  # 13
        True,  # 14
        True,  # 15
    )

    both_results = ~(~np.array(start_results) | ~np.array(end_results))

    ongoing_results = (
        False,  # 1
        False,  # 2
        False,  # 3
        False,  # 4
        False,  # 5
        False,  # 6
        False,  # 7
        True,  # 8
        True,  # 9
        False,  # 10
        False,  # 11
        False,  # 12
        False,  # 13
        False,  # 14
        True,  # 15
    )

    @pytest.fixture(
        params=(
            (
                {
                    "event_type": "event1",
                    "termination_mode": "ongoing",
                    "comment": "comment",
                },
                {
                    "event_type": "event1",
                    "termination_mode": "ongoing",
                    "comment": "comment",
                },
            ),
            (
                {"event_type": "event1"},
                {"event_type": "event1", "termination_mode": "ongoing"},
            ),
            (
                {"event_type": "event1", "termination_mode": "start"},
                {"event_type": "event1", "termination_mode": "start"},
            ),
            (
                {"event_type": "event1", "termination_mode": "end"},
                {"event_type": "event1", "termination_mode": "end"},
            ),
            (
                {"event_type": "event1", "termination_mode": "both"},
                {"event_type": "event1", "termination_mode": "both"},
            ),
            ({"event_type": "event1", "termination_mode": "something_wrong"}, ValueError),
        )
    )
    def init_data(self, request):
        return request.param

    @staticmethod
    def _test_check(strides_events, result, termination_mode):
        stride_start_stop, event_start_stop = strides_events
        c = EventTerminationCriteria(event_type="event1", termination_mode=termination_mode)
        stride_list = [window(*stride_start_stop[0]), window(*stride_start_stop[1])]
        events = [window(*event_start_stop)]
        event_list = [{"name": "event1", "events": events}]

        out = c.check_wb_start_end(
            stride_list, event_list=event_list, original_start=0, current_start=0, current_end=len(stride_list) - 1
        )

        result = len(stride_list) - 2 if bool(result) is False else None
        assert out[1] == result
        assert out[0] is None

    @pytest.mark.parametrize(("strides_events", "result"), (zip(test_cases, start_results)))
    def test_check_start_termination(self, strides_events, result):
        self._test_check(strides_events, result, "start")

    # TODO: Check moving start
    # TODO: Check event before the first stride

    @pytest.mark.parametrize(("strides_events", "result"), (zip(test_cases, end_results)))
    def test_check_end_termination(self, strides_events, result):
        self._test_check(strides_events, result, "end")

    @pytest.mark.parametrize(("strides_events", "result"), (zip(test_cases, both_results)))
    def test_check_both_termination(self, strides_events, result):
        self._test_check(strides_events, result, "both")

    @pytest.mark.parametrize(("strides_events", "result"), (zip(test_cases, ongoing_results)))
    def test_check_ongoing_termination(self, strides_events, result):
        self._test_check(strides_events, result, "ongoing")


class TestMaxTurnAngleCriteria(BaseTestCriteriaTermination):
    criteria_class = TurnAngleCriteria
    defaults = {"min_turn_angle": 0, "max_turn_angle": 1, "comment": "comment"}

    @pytest.fixture(
        params=(
            (
                {
                    "min_turn_angle": -1,
                    "max_turn_angle": 0,
                    "min_turn_rate": 0,
                    "max_turn_rate": 1,
                    "name": "name",
                    "comment": "comment",
                },
                {
                    "min_turn_angle": -1,
                    "max_turn_angle": 0,
                    "min_turn_rate": 0,
                    "max_turn_rate": 1,
                    "name": "name",
                    "comment": "comment",
                },
            ),
            (
                {"min_turn_angle": None, "max_turn_angle": 0},
                {"min_turn_angle": -np.inf, "max_turn_angle": 0},
            ),
            (
                {"min_turn_angle": 0, "max_turn_angle": None},
                {"min_turn_angle": 0, "max_turn_angle": np.inf},
            ),
            ({"min_turn_angle": 0, "max_turn_angle": 0}, ValueError),
            ({"min_turn_angle": 2, "max_turn_angle": 1}, ValueError),
            (
                {"min_turn_rate": None, "max_turn_rate": 0},
                {"min_turn_rate": -np.inf, "max_turn_rate": 0},
            ),
            (
                {"min_turn_rate": 0, "max_turn_rate": None},
                {"min_turn_rate": 0, "max_turn_rate": np.inf},
            ),
            ({"min_turn_rate": 0, "max_turn_rate": 0}, ValueError),
            ({"min_turn_rate": 2, "max_turn_rate": 1}, ValueError),
            (
                {},
                {
                    "min_turn_angle": -np.inf,
                    "max_turn_angle": np.inf,
                    "min_turn_rate": -np.inf,
                    "max_turn_rate": np.inf,
                    "name": None,
                    "comment": None,
                },
            ),
        )
    )
    def init_data(self, request):
        return request.param

    @pytest.mark.parametrize(
        ("strides_events", "result"),
        (
            zip(
                TestEventTerminationCriteria.test_cases,
                TestEventTerminationCriteria.ongoing_results,
            )
        ),
    )
    def test_check_ongoing(self, strides_events, result):
        TestEventTerminationCriteria._test_check(strides_events, result, "ongoing")

    def test_event_filter_angle(self):
        a = 10
        events_include = [window(0, 1, parameter={"angle": a}) for _ in range(10)]
        events_exclude = [window(0, 1, parameter={"angle": a + 10}) for _ in range(10)]
        events = [{"name": "turn", "events": [*events_include, *events_exclude]}]

        c = TurnAngleCriteria(a - 5, a + 5)
        assert c.filter_events(events) == events_include

        c = TurnAngleCriteria(a - 5, a + 20)
        assert c.filter_events(events) == events[0]["events"]

        c = TurnAngleCriteria(a - 10, a - 5)
        assert c.filter_events(events) == []

        c = TurnAngleCriteria(a + 11, a + 20)
        assert c.filter_events(events) == []

    def test_event_filter_rate(self):
        a = 10
        events_include = [window(0, 10, parameter={"angle": a}) for _ in range(10)]  # -> Turnrate 1
        events_exclude = [window(0, 5, parameter={"angle": a}) for _ in range(10)]  # -> Turnrate 2
        events = [{"name": "turn", "events": [*events_include, *events_exclude]}]

        c = TurnAngleCriteria(min_turn_rate=0, max_turn_rate=1.5)
        assert c.filter_events(events) == events_include

        c = TurnAngleCriteria()
        assert c.filter_events(events) == events[0]["events"]

        c = TurnAngleCriteria(min_turn_rate=0, max_turn_rate=1)
        assert c.filter_events(events) == events_include

        c = TurnAngleCriteria(min_turn_rate=1, max_turn_rate=1.5)
        assert c.filter_events(events) == events_include


class TestLevelWalkingCriteria(BaseTestCriteriaTermination):
    criteria_class = LevelWalkingCriteria
    defaults = {
        "max_non_level_strides": 1,
        "max_non_level_strides_left": 2,
        "max_non_level_strides_right": 3,
        "level_walking_threshold": 4,
        "name": "name",
        "comment": "comment",
    }

    @pytest.fixture(
        params=(
            (
                {
                    "max_non_level_strides": 1,
                    "max_non_level_strides_left": 2,
                    "max_non_level_strides_right": 3,
                    "level_walking_threshold": 4,
                    "name": "name",
                    "comment": "comment",
                },
                {
                    "max_non_level_strides": 1,
                    "max_non_level_strides_left": 2,
                    "max_non_level_strides_right": 3,
                    "level_walking_threshold": 4,
                    "name": "name",
                    "comment": "comment",
                },
            ),
            (
                {
                    "max_non_level_strides": 1,
                    "max_non_level_strides_left": 2,
                    "max_non_level_strides_right": 3,
                    "level_walking_threshold": -3,
                    "name": "name",
                    "comment": "comment",
                },
                ValueError,
            ),
        )
    )
    def init_data(self, request):
        return request.param

    @pytest.mark.skip(reason="Not applicable for this criteria")
    def test_single_stride(self, naive_stride_list, naive_event_list):
        pass

    @pytest.mark.parametrize(
        ("invalid_strides", "expected_end"),
        (
            ([4, 5, 6], 4),
            ([4, 5], 20),
            ([4, 5, 6, 7], 4),
            ([4, 5, 7, 8], 20),
            ([4, 5, 7, 8, 9], 7),
        ),
    )
    def test_simple_break(self, invalid_strides, expected_end):
        lr = repeat(("left", "right"))
        thres = 2
        strides = [window(start=s, end=s + 1, foot=next(lr), parameter={"elevation": 0}) for s in range(20)]
        for i, s in enumerate(strides):
            if i in invalid_strides:
                s["parameter"]["elevation"] = thres + 1

        # Test directly in combination with the WBA
        wba = WBAssembly([("level_walk", LevelWalkingCriteria(max_non_level_strides=3, level_walking_threshold=thres))])
        wba.assemble(strides)

        assert expected_end == wba.wb_list_[0]["end"]

    def test_combined_break_rules(self):
        """Test multiple delayed rules.

        This test checks what happens if multiple delay rules are active at the same time.
        """
        # First rule fires earlier with higher threshold
        r1 = LevelWalkingCriteria(max_non_level_strides=2, level_walking_threshold=2)
        # Second rule has higher delay but lower threshold
        r2 = LevelWalkingCriteria(max_non_level_strides=4, level_walking_threshold=1)

        test_elevation = [0, 0, 0, 0, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5]
        lr = repeat(("left", "right"))
        strides = [
            window(start=s, end=s + 1, foot=next(lr), parameter={"elevation": test_elevation[s]})
            for s in range(len(test_elevation))
        ]

        # only rule one
        wba = WBAssembly([("r1", r1)])
        wba.assemble(strides)

        assert wba.wb_list_[0]["end"] == 7

        # only rule two
        wba = WBAssembly([("r2", r2)])
        wba.assemble(strides)

        assert wba.wb_list_[0]["end"] == 4

        # combined rules
        wba = WBAssembly([("r1", r1), ("r2", r2)])
        wba.assemble(strides)

        assert wba.wb_list_[0]["end"] == 4
