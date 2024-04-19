from typing import Any

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack


class BaseTurnDetector(Algorithm):
    _action_methods = ("detect",)

    # Other Parameters
    data: pd.DataFrame
    sampling_rate_hz: float

    # results
    turn_list_: pd.DataFrame

    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **kwargs: Unpack[dict[str, Any]]) -> Self:
        raise NotImplementedError


__all__ = ["BaseTurnDetector"]
