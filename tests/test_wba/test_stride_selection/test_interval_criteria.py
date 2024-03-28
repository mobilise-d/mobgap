import pandas as pd
import pytest

from mobgap.wba import IntervalDurationCriteria, IntervalParameterCriteria
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

    @pytest.mark.parametrize("para_value", (-1, 0, 40, 100))
    @pytest.mark.parametrize("lower_upper", (0, 1))
    def test_inclusive(self, lower_upper, para_value):
        interval = pd.Series(window(0, 0, para=para_value))
        rule = IntervalParameterCriteria(
            "para",
            para_value if lower_upper == 0 else None,
            para_value if lower_upper == 1 else None,
            inclusive=(False, False),
        )
        # First we test that it is rejected
        assert not rule.check(interval)

        # Now we test that it is accepted
        inclusive_value = (True, False) if lower_upper == 0 else (False, True)
        rule.set_params(inclusive=inclusive_value)

        assert rule.check(interval) is True

        # Now we check that it is true if both are true
        rule.set_params(inclusive=(True, True))

        assert rule.check(interval) is True


class TestIntervalDurationCriteria:
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
        c = IntervalDurationCriteria(lower, upper)
        test_stride = pd.Series(window(0, value))

        assert c.check(test_stride, sampling_rate_hz=1) == result

    @pytest.mark.parametrize("duration", (1, 0, 40, 100))
    @pytest.mark.parametrize("lower_upper", (0, 1))
    def test_inclusive(self, lower_upper, duration):
        interval = pd.Series(window(0, duration))
        rule = IntervalDurationCriteria(
            duration if lower_upper == 0 else None,
            duration if lower_upper == 1 else None,
            inclusive=(False, False),
        )
        # First we test that it is rejected
        assert not rule.check(interval, sampling_rate_hz=1)

        # Now we test that it is accepted
        inclusive_value = (True, False) if lower_upper == 0 else (False, True)
        rule.set_params(inclusive=inclusive_value)

        assert rule.check(interval, sampling_rate_hz=1) is True

        # Now we check that it is true if both are true
        rule.set_params(inclusive=(True, True))

        assert rule.check(interval, sampling_rate_hz=1) is True
