from typing import Optional, Union

import pandas as pd
from typing_extensions import Literal

from mobgap.wba._wb_criteria_base import (
    BaseWbCriteria,
)


class NStridesCriteria(BaseWbCriteria):
    """Min number of strides in the WB.

    If any of the three criteria is True the WB is accepted.
    """

    min_strides: Optional[int]
    min_strides_left: Optional[int]
    min_strides_right: Optional[int]

    _FOOT_COL_NAME: str = "foot"

    def __init__(
        self,
        min_strides: Optional[int] = None,
        min_strides_left: Optional[int] = None,
        min_strides_right: Optional[int] = None,
    ) -> None:
        self.min_strides = min_strides
        self.min_strides_left = min_strides_left
        self.min_strides_right = min_strides_right

    def check_include(
        self,
        stride_list: pd.DataFrame,
        *,
        sampling_rate_hz: Optional[float] = None,  # noqa: ARG002
    ) -> bool:
        if self.min_strides is not None:
            if self.min_strides < 0:
                raise ValueError(f"Only positive values are allowed for `min_strides` not {self.min_strides}")
            return len(stride_list) >= self.min_strides
        if self.min_strides_left is None and self.min_strides_right is None:
            return False
        foot = stride_list[self._FOOT_COL_NAME]
        foot_count = foot.value_counts()
        if self.min_strides_left is not None:
            if self.min_strides_left < 0:
                raise ValueError(f"Only positive values are allowed for `min_strides_left` not {self.min_strides_left}")
            return foot_count.get("left", -1) >= self.min_strides_left
        if self.min_strides_right is not None:
            if self.min_strides_right < 0:
                raise ValueError(
                    f"Only positive values are allowed for `min_strides_right` not {self.min_strides_right}"
                )
            return foot_count.get("right", -1) >= self.min_strides_right
        return False


class MaxBreakCriteria(BaseWbCriteria):
    """Test if the break between the last two strides of a window list is larger than a threshold.

    Parameters
    ----------
    max_break_s
        The maximal allowed break between two strides independent of the foot.
        It will be compared with <=.
        The unit depends on the unit used in the stride list that is filtered.
    remove_last_ic
        Because the last initial contact each foot in a WB, are no real initial contacts (they are not the start of a
        new stride), it might be advisable to remove the last stride from a WB when it was terminated by a break.
        If `remove_last_ic` is True, the last stride will be removed from the WB.
        If `remove_last_ic` is "per_foot", the last stride of each foot will be removed, if the last two strides were
        performed with different feet.
        In case they were performed with the same feet, it is assumed that the last stride of the other foot was missed
        in the recording and hence, only the last stride will be removed.
    consider_end_as_break
        If True, the termination rule will fire at the very end of the stride list.
        I.e. it will consider the end of the stride list as a break.
        This should be set to True, if `remove_last_ic` is used, as otherwise the post-processing will not happen for
        the last bout in the stride list.

    """

    max_break_s: float

    _FOOT_COL_NAME: str = "foot"
    _START_COL_NAME: str = "start"
    _END_COL_NAME: str = "end"

    def __init__(
        self,
        max_break_s: float,
        remove_last_ic: Union[bool, Literal["per_foot"]] = False,
        consider_end_as_break: bool = True,
    ) -> None:
        self.max_break_s = max_break_s
        self.remove_last_ic = remove_last_ic
        self.consider_end_as_break = consider_end_as_break

    def check_wb_start_end(
        self,
        stride_list: pd.DataFrame,
        *,
        original_start: int,
        current_start: int,  # noqa: ARG002
        current_end: int,
        sampling_rate_hz: Optional[float] = None,
    ) -> tuple[Optional[int], Optional[int], Optional[int]]:
        if sampling_rate_hz is None:
            raise ValueError("The sampling rate must be provided if the MaxBreakCriteria is used.")

        if self.max_break_s < 0:
            raise ValueError(f'Only positive values are allowed for "max_break" not {self.max_break_s}')

        if not isinstance(self.remove_last_ic, bool) and not self.remove_last_ic == "per_foot":
            raise ValueError("`remove_last_ic` must be a Boolean or the string 'per_foot'.")

        # The method is called once including the final stride.
        # We handle this case only, if the `consider_end_as_break` is True.
        if current_end == len(stride_list):
            if self.consider_end_as_break is True:
                return self._process_break(stride_list, original_start, current_end)
            return None, None, None

        if current_end - original_start < 1:
            return None, None, None

        last_stride = stride_list.iloc[current_end - 1]
        current_stride = stride_list.iloc[current_end]

        if (
            current_stride[self._START_COL_NAME] - last_stride[self._END_COL_NAME]
        ) / sampling_rate_hz <= self.max_break_s:
            # No break -> no termination
            return None, None, None
        # Break -> terminate
        return self._process_break(stride_list, original_start, current_end)

    def _process_break(
        self,
        stride_list: pd.DataFrame,
        original_start: int,
        current_end: int,
    ) -> tuple[Optional[int], Optional[int], Optional[int]]:
        # This means the current stride is not part of the WB
        # The last stride is at index current_end - 1
        wb_end = current_end - 1
        # While we remove strides from the end, we don't want to try to start a new WB.
        if self.remove_last_ic is True:
            wb_end -= 1
        elif self.remove_last_ic == "per_foot":
            # TODO: Add proper tests for this
            # For this approach we need at least 2 strides in the WB
            if wb_end - original_start < 2:
                # If we don't have that we basically just remove this stride and remove the WB
                # I am not sure if we even can end up here, but just in case
                return None, original_start, current_end
            # If the last two strides of the terminated wb have different feet values remove them both. If they have
            # the same, remove only the last, as we assume that the IC of the other foot was not detected
            feet = stride_list[self._FOOT_COL_NAME]
            second_to_last_foot = feet.iloc[wb_end - 1]
            last_foot = feet.iloc[wb_end]
            if last_foot and second_to_last_foot and last_foot != second_to_last_foot:
                wb_end -= 2
            else:
                # The last two strides are from the same foot.
                # We assume we did not correctly detect the second to last stride and hence, only remove the last stride
                # and not the last two strides.
                wb_end -= 1
        return None, wb_end, current_end


class LeftRightCriteria(BaseWbCriteria):
    """Test a left stride is always followed by a right stride.

    The WB is broken if two consecutive strides are performed with the same foot.
    """

    max_break: float

    _FOOT_COL_NAME: str = "foot"

    def check_wb_start_end(
        self,
        stride_list: pd.DataFrame,
        *,
        original_start: int,  # noqa: ARG002
        current_start: int,  # noqa: ARG002
        current_end: int,
        sampling_rate_hz: Optional[float] = None,  # noqa: ARG002
    ) -> tuple[Optional[int], Optional[int], Optional[int]]:
        if current_end < 1:
            return None, None, None
        feet = stride_list[self._FOOT_COL_NAME]
        last_foot = feet.iloc[current_end - 1]
        this_foot = feet.iloc[current_end]
        if last_foot and this_foot and last_foot == this_foot:
            return None, current_end - 1, current_end - 1
        return None, None, None
