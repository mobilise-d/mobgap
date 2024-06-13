import warnings
from typing import Literal, Union

import numpy as np
import pandas as pd


def _handle_zero_division(
    divisor: Union[pd.Series, pd.DataFrame],
    zero_division_hint: Union[Literal["warn", "raise"], float],
    caller_fct_name: str,
) -> None:
    if (divisor == 0).any():
        if zero_division_hint not in ["warn", "raise", np.nan]:
            raise ValueError('"zero_division" must be set to "warn", "raise" or `np.nan`!')
        if zero_division_hint == "raise":
            raise ZeroDivisionError(f"Zero division occurred in {caller_fct_name} because divisor contains zeroes.")
        if zero_division_hint == "warn":
            warnings.warn(
                f"Zero division occurred in {caller_fct_name} because divisor contains zeroes. "
                "Affected error metrics are set to NaN.",
                UserWarning,
                stacklevel=2,
            )
