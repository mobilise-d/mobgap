from typing import Optional

import pandas as pd
from tpcp import BaseTpcpObject

from mobgap.wba._utils import compare_with_threshold


class BaseWbCriteria(BaseTpcpObject):
    def check_wb_start_end(
        self,
        stride_list: pd.DataFrame,  # noqa: ARG002
        *,
        original_start: int,  # noqa: ARG002
        current_start: int,  # noqa: ARG002
        current_end: int,
        sampling_rate_hz: Optional[float] = None,  # noqa: ARG002
    ) -> tuple[Optional[int], Optional[int], Optional[int]]:
        """Determine the current start and end of the current WB.

        This method is expected to return 3 values:

            - The adjusted start index (or None if no adjustment is required)
            - The adjusted end index (or None if no adjustment is required)
            - The index the WBA should restart the search after the termination.
              This is often simply `current_end`, but can be adjusted if you want to allow a new search to start
              earlier.
              This might be useful in cases, where the end index was adjusted, but starting a new search directly after
              the adjusted end would still be meaningfull.

        This method gets passed all strides (past and future) of the entire measurement.
        All this information can be used in the rule.
        However, looking in the future is discouraged, as this might interfere with other rules.

        The method will be called for every end-stride the WBA considers.
        The method needs to decide, if the new stride (`stride_list.iloc[current_end]`) should be included in the
        WB or not.
        If the stride should be included, the method should return `None, None, current_end`.
        If the stride should not be included, the method should return the adjusted end indices (second value).
        For example, `None, current_end - 1, current_end` would mean that the stride should not be included and the
        search for a new WB will start with the stride that was rejected.
        However, the method can also decide that given the current strides, the WB should be terminated even earlier.
        In this case, the method can return an arbitrary index as the second value.
        For example, `None, current_end - 2, current_end` would mean that the current stride is not part of the WB and
        the last stride is actively excluded from the WB.
        However, the search is still restarted with the current stride as the third value is set to `current_end`.
        This avoids restarting a new WB, that (depending on the rule), would be terminated immediately again.
        Sometimes, it makes sense to set the last value to something else than `current_end`.
        This would allow to start a new WB earlier or even skip a number of strides, before a new WB can be started.
        Only very complex rules usually need this.

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
        sampling_rate_hz
            The sampling rate of the data in Hz.
            This is used to potentially convert provided stride value to seconds, assuming that they are in
            samples.

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
        restart_index
            The index at which the WBA should restart the search for a new WB.
            By default, this should be `current_end`, but can be adjusted if the rule wants to allow a new search to
            start earlier or even later (skipping a number of strides).
            This value is only considered, if the respective WB-rule is used for the termination of the WB.

        """
        return None, None, current_end

    def check_include(self, preliminary_wb: dict, *, sampling_rate_hz: Optional[float] = None) -> bool:  # noqa: ARG002
        """Check if a preliminary WB should be considered an actual WB.

        Parameters
        ----------
        preliminary_wb
            The preliminary wb including its stride list
        sampling_rate_hz
            The sampling rate of the data in Hz.
            This is used to potentially convert values of a stride to seconds if required, assuming that they are in
            samples.
            If this is not the case and the values are already in seconds, `sampling_rate_hz` should set to 1.

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

    def check_include(
        self,
        wb: dict,
        sampling_rate_hz: Optional[float] = None,
    ) -> bool:
        stride_list = wb["strideList"]
        value = self._calc_summary(stride_list, sampling_rate_hz)
        return compare_with_threshold(value, self.lower_threshold, self.upper_threshold, self.inclusive)

    def _calc_summary(self, stride_list: pd.DataFrame, sampling_rate_hz: Optional[float]) -> float:
        raise NotImplementedError("This needs to implemented by child class")


class EndOfList(BaseWbCriteria):
    """Dummy criteria to describe the end of the stride list.

    DO NOT USE THIS AS A CUSTOM RULE
    """
