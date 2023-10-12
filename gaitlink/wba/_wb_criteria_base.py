from typing import Optional

import pandas as pd
from tpcp import BaseTpcpObject

from gaitlink.wba._utils import compare_with_threshold


class BaseWbCriteria(BaseTpcpObject):
    def check_wb_start_end(
        self,
        stride_list: pd.DataFrame,  # noqa: ARG002
        *,
        original_start: int,  # noqa: ARG002
        current_start: int,  # noqa: ARG002
        current_end: int,  # noqa: ARG002
    ) -> tuple[Optional[int], Optional[int]]:
        """Determine the current start and end of the current WB.

        This method gets passed all strides (past and future) of the entire measurement.
        All this information can be used in the rule.
        However, looking in the future is discouraged, as this might interfere with other rules.

        Parameters
        ----------
        stride_list
            A list of all strides within the measurement.
        original_start
            The index in the stride list at which the WB was originally started.
            This is usually the stride after the end of the last WB.
        current_start
            The index at which the WB actually starts based on all rules so far
        current_end
            The index at which the WB is supposed to end.
            This is always the index of the stride that should be considered for the WB.

        Returns
        -------
        adjusted_start
            An optional index of the adjusted start value.
            If multiple rules adjust the same start value in the same iteration, the one that predicts the latest start
            is used.
            If None is returned, this is equivalent to not adjusting the start value and should be the default.
        adjusted_end
            An optional index of the adjusted end value.
            If the rule predict that the WB should be terminated - excluding the current stride (current_end) - it
            should return `current_end - 1`.
            If earlier termination times are required the respective index should be returned.
            In case multiple rules predict a termination, the one with the earliest index is chosen.
            If the WB should not be terminated either None or `current_end` should be returned.
        """
        return None, None

    def check_include(self, preliminary_wb: dict) -> bool:  # noqa: ARG002
        """Check if a preliminary WB should be considered an actual WB.

        Parameters
        ----------
        preliminary_wb
            The preliminary wb including its stride list

        Returns
        -------
        is_actual_wb
            True or False if the preliminary WB should be considered an actual WB
        """
        return True


class BaseSummaryCriteria(BaseWbCriteria):
    lower_threshold: float
    upper_threshold: float
    inclusive: tuple[bool, bool]

    def __init__(
        self,
        lower_threshold: Optional[float] = None,
        upper_threshold: Optional[float] = None,
        *,
        inclusive: tuple[bool, bool] = (False, True),
    ) -> None:
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.inclusive = inclusive

    def check_include(self, wb: dict) -> bool:
        stride_list = wb["strideList"]
        value = self._calc_summary(stride_list)
        return compare_with_threshold(value, self.lower_threshold, self.upper_threshold, self.inclusive)

    def _calc_summary(self, stride_list: pd.DataFrame) -> float:
        raise NotImplementedError("This needs to implemented by child class")


class EndOfList(BaseWbCriteria):
    """Dummy criteria to describe the end of the stride list.

    DO NOT USE THIS AS A CUSTOM RULE
    """
