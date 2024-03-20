import numpy as np
import pytest

from mobgap.wba._utils import check_thresholds


@pytest.mark.parametrize(
    ("lower", "upper", "allow_both_none"),
    (
        (0, 0, False),
        (np.inf, None, False),
        (None, -np.inf, False),
        (None, None, False),
        (-2, -3, None),
        (5.0, 3.0, None),
    ),
)
def test_lower_upper_invalid(lower, upper, allow_both_none):
    with pytest.raises(ValueError):
        check_thresholds(lower, upper, allow_both_none)


@pytest.mark.parametrize(
    ("inputs", "out"),
    (
        ((None, None, True), (-np.inf, np.inf)),
        ((None, np.inf, False), (-np.inf, np.inf)),
        ((-np.inf, None, False), (-np.inf, np.inf)),
        ((-3, -2, False), (-3, -2)),
        ((3, 5, False), (3, 5)),
    ),
)
def test_lower_upper_valid(inputs, out):
    assert check_thresholds(*inputs) == out
