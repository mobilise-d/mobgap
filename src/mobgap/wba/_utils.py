from typing import Optional

import numpy as np
import pandas as pd


def check_thresholds(
    lower_threshold: Optional[float] = None,
    upper_threshold: Optional[float] = None,
    allow_both_none: bool = False,
) -> tuple[float, float]:
    if allow_both_none is False and lower_threshold is None and upper_threshold is None:
        raise ValueError("You need to provide at least a upper or a lower threshold.")
    if lower_threshold is None:
        lower_threshold = -np.inf
    if upper_threshold is None:
        upper_threshold = np.inf

    if not lower_threshold < upper_threshold:
        raise ValueError(
            "The lower threshold must be below the upper threshold. "
            f"Currently: {lower_threshold} not < {upper_threshold}"
        )
    return lower_threshold, upper_threshold


def compare_with_threshold(
    value: float,
    lower_threshold: float,
    upper_threshold: float,
    inclusive: tuple[bool, bool],
) -> bool:
    lower_threshold, upper_threshold = check_thresholds(lower_threshold, upper_threshold)
    # Lower comparison
    operator = np.greater_equal if inclusive[0] else np.greater
    lower_comparison = operator(value, lower_threshold)
    # Upper comparison
    operator = np.less_equal if inclusive[1] else np.less
    upper_comparison = operator(value, upper_threshold)
    # We convert to bool, so that we don't have to deal with numpy dtypes
    return bool(lower_comparison and upper_comparison)


def compare_with_threshold_multiple(
    values: pd.Series,
    lower_threshold: float,
    upper_threshold: float,
    inclusive: tuple[bool, bool],
) -> pd.Series:
    lower_threshold, upper_threshold = check_thresholds(lower_threshold, upper_threshold)
    # Lower comparison
    operator = np.greater_equal if inclusive[0] else np.greater
    lower_comparison = operator(values, lower_threshold)
    # Upper comparison
    operator = np.less_equal if inclusive[1] else np.less
    upper_comparison = operator(values, upper_threshold)
    return pd.Series((lower_comparison & upper_comparison).astype(bool), index=values.index)
