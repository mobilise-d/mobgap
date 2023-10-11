import pytest

from gaitlink.wba._wb_assembly import create_wb
from gaitlink.wba._wb_criteria import (
    LeftRightCriteria,
    MaxBreakCriteria,
    NStridesCriteria,
)
from gaitlink.wba._wb_criteria_base import BaseWBCriteria
from tests.test_wba.conftest import window


class BaseTestCriteriaInclusion:
    criteria_class: type[BaseWBCriteria]

    defaults = {}

    def test_single_stride(self, naive_stride_list, naive_event_list):
        """Test that now exception is thrown if a list with a single stride is checked."""
        single_stride_list = [naive_stride_list[0]]

        self.criteria_class(**self.defaults).check_include(create_wb(single_stride_list), naive_event_list)


class BaseTestCriteriaTermination:
    criteria_class: type[BaseWBCriteria]

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
