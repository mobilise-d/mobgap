from typing import Optional

import pandas as pd
from tpcp import BaseTpcpObject

from gaitlink.wba._utils import check_thresholds


class IntervalSummaryCriteria(BaseTpcpObject):
    comment: Optional[str]

    def __init__(self, comment: Optional[str] = None) -> None:
        self.comment = comment

    def check(self, interval: pd.Series) -> bool:
        raise NotImplementedError("This needs to implemented by child class")


class IntervalParameterCriteria(IntervalSummaryCriteria):
    lower_threshold: float
    upper_threshold: float
    parameter: str

    def __init__(
        self,
        parameter: str,
        lower_threshold: Optional[float] = None,
        upper_threshold: Optional[float] = None,
        *,
        comment: Optional[str] = None,
    ) -> None:
        lower_threshold, upper_threshold = check_thresholds(lower_threshold, upper_threshold)

        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.parameter = parameter

        super().__init__(comment=comment)

    def check(self, interval: pd.Series) -> bool:
        return self.lower_threshold < interval[self.parameter] <= self.upper_threshold


class IntervalDurationCriteria(IntervalSummaryCriteria):
    """Checks the duration of the stride by subtracting the start and the end timestamp."""

    lower_threshold: float
    upper_threshold: float

    start_col_name: str
    end_col_name: str

    def __init__(
        self,
        lower_threshold: Optional[float] = None,
        upper_threshold: Optional[float] = None,
        *,
        start_col_name: str = "start",
        end_col_name: str = "end",
        comment: Optional[str] = None,
    ) -> None:
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

        self.start_col_name = start_col_name
        self.end_col_name = end_col_name

        super().__init__(comment=comment)

    def check(self, interval: pd.Series) -> bool:
        # TODO: Test
        lower_threshold, upper_threshold = check_thresholds(self.lower_threshold, self.upper_threshold)

        duration = interval[self.end_col_name] - interval[self.start_col_name]
        return lower_threshold < duration <= upper_threshold
