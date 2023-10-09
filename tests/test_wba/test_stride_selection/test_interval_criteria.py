import pandas as pd
import pytest

from gaitlink.wba._interval_criteria import IntervalParameterCriteria
from tests.test_wba.conftest import window


class TestIntervalParameterCriteria:
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
        c = IntervalParameterCriteria("length", lower, upper)
        test_stride = pd.Series(window(0, 0, length=value))

        assert c.check(test_stride) == result
