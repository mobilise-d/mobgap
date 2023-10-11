from collections import Counter
from typing import Optional, Union

from typing_extensions import Literal

from gaitlink.wba._wb_criteria_base import (
    BaseWBCriteria,
)


class NStridesCriteria(BaseWBCriteria):
    """Min number of strides in the WB.

    If any of the three criteria is True the WB is accepted.
    """

    min_strides: Optional[int]
    min_strides_left: Optional[int]
    min_strides_right: Optional[int]

    def __init__(
        self,
        min_strides: Optional[int] = None,
        min_strides_left: Optional[int] = None,
        min_strides_right: Optional[int] = None,
    ) -> None:
        self.min_strides = min_strides
        self.min_strides_left = min_strides_left
        self.min_strides_right = min_strides_right

    def check_include(self, wb: dict) -> bool:
        stride_list = wb["strideList"]
        if self.min_strides is not None:
            return len(stride_list) >= self.min_strides
        if self.min_strides_left is None and self.min_strides_right is None:
            return False
        foot = [s["foot"] for s in stride_list]
        foot_count = Counter(foot)
        if self.min_strides_left is not None:
            return foot_count["left"] >= self.min_strides_left
        if self.min_strides_right is not None:
            return foot_count["right"] >= self.min_strides_right
        return False


class MaxBreakCriteria(BaseWBCriteria):
    """Test if the break between the last two strides of a window list is larger than a threshold.

    Parameters
    ----------
    max_break
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
    """

    max_break: float

    def __init__(
        self,
        max_break: float,
        remove_last_ic: Union[bool, Literal["per_foot"]] = False,
    ) -> None:
        if max_break < 0:
            raise ValueError(f'Only positive values are allowed for "max_break" not {max_break}')
        self.max_break = max_break
        if not isinstance(remove_last_ic, bool) and not remove_last_ic == "per_foot":
            raise ValueError("`remove_last_ic` must be a Boolean or the string 'per_foot'.")
        self.remove_last_ic = remove_last_ic

    def check_wb_start_end(
        self,
        stride_list: list[dict],
        original_start: int,
        current_start: int,
        current_end: int,
    ) -> tuple[Optional[int], Optional[int]]:
        if current_end - original_start < 1:
            return None, None
        last_stride = stride_list[current_end - 1]
        current_stride = stride_list[current_end]
        if current_stride["start"] - last_stride["end"] <= self.max_break:
            return None, None
        lag = 0
        if self.remove_last_ic is True:
            lag = 1
        elif self.remove_last_ic == "per_foot":
            # If the last two strides of the terminated wb have different feet values remove them both. If they have
            # the same, remove only the last, as we assume that the IC of the other foot was not detected
            lag = 1
            if len(stride_list) >= 3:
                last_foot = stride_list[-3]["foot"]
                this_foot = stride_list[-2]["foot"]
                if last_foot and this_foot and last_foot != this_foot:
                    lag = 2
        return None, current_end - lag - 1


class LeftRightCriteria(BaseWBCriteria):
    """Test a left stride is always followed by a right stride.

    The WB is broken if two consecutive strides are performed with the same foot.
    """

    max_break: float

    def check_wb_start_end(
        self,
        stride_list: list[dict],
        original_start: int,
        current_start: int,
        current_end: int,
    ) -> tuple[Optional[int], Optional[int]]:
        if current_end < 1:
            return None, None
        last_foot = stride_list[current_end - 1]["foot"]
        this_foot = stride_list[current_end]["foot"]
        if last_foot and this_foot and last_foot == this_foot:
            return None, current_end - 1
        return None, None
