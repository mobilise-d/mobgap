import pytest

from gaitlink.wba._stride_criteria import ThresholdCriteria
from tests.test_wba.conftest import window


class TestThresholdCriteria:
    @pytest.mark.parametrize(
        ("lower", "upper", "value", "result"),
        (
            (0, 1, 0.5, True),
            (0, 1, 0, False),
            (0, 1, 1, True),
            (0, 1, 1.2, False),
            (0, 1, -1, False),
            (None, 1, -1, True),
            (0, None, 1, True),
        ),
    )
    def test_check_stride(self, lower, upper, value, result):
        c = ThresholdCriteria("length", lower, upper)
        test_stride = window(0, 0, parameter={"length": value})

        assert c.check(test_stride) == result

    @pytest.mark.parametrize(
        ("lower", "upper", "value", "result"),
        (
            (0, 1, 0.5, True),
            (0, 1, 0, False),
            (0, 1, 1, True),
            (0, 1, 1.2, False),
            (0, 1, -1, False),
            (None, 1, -1, True),
            (0, None, 1, True),
        ),
    )
    def test_check_stride_list(self, lower, upper, value, result):
        c = ThresholdCriteria("length", lower, upper)
        test_stride_list = [window(0, 0, parameter={"length": value}) for _ in range(10)]
        assert c.check_stride_list(test_stride_list) == result
