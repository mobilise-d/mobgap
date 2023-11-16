from typing import Unpack, Any

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self


class BaseCadenceCalculator(Algorithm):
    data: pd.DataFrame
    initial_contacts: pd.Series

    def calculate(
        self,
        data: pd.DataFrame,
        initial_contacts: pd.Series,
        *,
        sampling_rate_hz: float,
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        raise NotImplementedError
