from typing import Optional

import numpy as np


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
            "The lower threshold must be below the upper threshold. Currently: {} not < {}".format(
                lower_threshold, upper_threshold
            )
        )
    return lower_threshold, upper_threshold
