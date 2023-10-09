from collections import Counter
from typing import Optional, Union

import numpy as np
from typing_extensions import Literal

from gaitlink.wba._utils import check_thresholds
from gaitlink.wba._wb_criteria_base import (
    EventTerminationCriteria,
    WBCriteria,
)


class NStridesCriteria(WBCriteria):
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
        comment: Optional[str] = None,
    ) -> None:
        self.min_strides = min_strides
        self.min_strides_left = min_strides_left
        self.min_strides_right = min_strides_right

        super().__init__(comment=comment)

    def check_include(self, wb: dict, event_list: Optional[list[dict]] = None) -> bool:
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


class MaxBreakCriteria(WBCriteria):
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
        comment: Optional[str] = None,
    ) -> None:
        if max_break < 0:
            raise ValueError(f'Only positive values are allowed for "max_break" not {max_break}')
        self.max_break = max_break
        if not isinstance(remove_last_ic, bool) and not remove_last_ic == "per_foot":
            raise ValueError("`remove_last_ic` must be a Boolean or the string 'per_foot'.")
        self.remove_last_ic = remove_last_ic

        super().__init__(comment=comment)

    def check_wb_start_end(
        self,
        stride_list: list[dict],
        original_start: int,
        current_start: int,
        current_end: int,
        event_list: Optional[list[dict]] = None,
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


class LeftRightCriteria(WBCriteria):
    """Test a left stride is always followed by a right stride.

    The WB is broken if two consecutive strides are performed with the same foot.
    """

    max_break: float

    _serializable_paras = ()

    _rule_type: str = "left_right_consistency"

    def check_wb_start_end(
        self,
        stride_list: list[dict],
        original_start: int,
        current_start: int,
        current_end: int,
        event_list: Optional[list[dict]] = None,
    ) -> tuple[Optional[int], Optional[int]]:
        if current_end < 1:
            return None, None
        last_foot = stride_list[current_end - 1]["foot"]
        this_foot = stride_list[current_end]["foot"]
        if last_foot and this_foot and last_foot == this_foot:
            return None, current_end - 1
        return None, None


class LevelWalkingCriteria(WBCriteria):
    """Test if WB has no more than N consecutive non-level strides.

    A WB is terminated if there are more than N consecutive strides that are not level walking.
    """

    _rule_type = "level_walking_criteria"

    max_non_level_strides: Optional[int]
    max_non_level_strides_left: Optional[int]
    max_non_level_strides_right: Optional[int]
    level_walking_threshold: float

    _serializable_paras = (
        "max_non_level_strides",
        "max_non_level_strides_left",
        "max_non_level_strides_right",
        "level_walking_threshold",
    )

    @property
    def _max_lag(self) -> int:
        non_none_vals = [
            val
            for val in [self.max_non_level_strides, self.max_non_level_strides_left, self.max_non_level_strides_right]
            if val
        ]
        return max(non_none_vals)

    def __init__(
        self,
        level_walking_threshold: float,
        max_non_level_strides: Optional[int] = None,
        max_non_level_strides_left: Optional[int] = None,
        max_non_level_strides_right: Optional[float] = None,
        field_name: str = "elevation",
        comment: Optional[str] = None,
    ) -> None:
        self.max_non_level_strides = max_non_level_strides
        self.max_non_level_strides_left = max_non_level_strides_left
        self.max_non_level_strides_right = max_non_level_strides_right
        if level_walking_threshold < 0:
            raise ValueError("`level_walking_threshold` must be >0.")
        self.level_walking_threshold = level_walking_threshold
        self.field_name = field_name
        super().__init__(comment=comment)

    def check_wb_start_end(
        self,
        stride_list: list[dict],
        original_start: int,
        current_start: int,
        current_end: int,
        event_list: Optional[list[dict]] = None,
    ) -> tuple[Optional[int], Optional[int]]:
        past_strides = stride_list[original_start : current_end + 1]
        consecutive_section = []
        for stride in reversed(past_strides):
            # We consider nan values always as level  walking!
            stride_height_change = abs(stride["parameter"][self.field_name])
            if not np.isnan(stride_height_change) and stride_height_change >= self.level_walking_threshold:
                consecutive_section.insert(0, stride)
            else:
                break
        if not consecutive_section:
            return None, None

        is_non_level = self._check_subsequence(consecutive_section)
        if is_non_level:
            # If we are at the beginning of the WB, we will change the start.
            if len(consecutive_section) == len(past_strides):
                return current_end + 1, None

            # If we are in the middle of a WB, we want to terminate it
            return None, current_end - len(consecutive_section)

        return None, None

    def _check_subsequence(self, stride_list) -> bool:
        """Check if the detected part exceeds our thresholds."""
        if self.max_non_level_strides is not None:
            return len(stride_list) >= self.max_non_level_strides
        if self.max_non_level_strides_left is None and self.max_non_level_strides_right is None:
            return False
        foot = [s["foot"] for s in stride_list]
        foot_count = Counter(foot)
        if self.max_non_level_strides_left is not None:
            return foot_count["left"] >= self.max_non_level_strides_left
        if self.max_non_level_strides_right is not None:
            return foot_count["right"] >= self.max_non_level_strides_right
        return False


class TurnAngleCriteria(EventTerminationCriteria):
    event_type: str = "turn"
    min_turn_angle: float
    max_turn_angle: float
    min_turn_rate: float
    max_turn_rate: float

    _serializable_paras = (
        "min_turn_angle",
        "max_turn_angle",
        "min_turn_rate",
        "max_turn_rate",
    )

    _rule_type: str = "turn_event"

    def __init__(
        self,
        min_turn_angle: Optional[float] = None,
        max_turn_angle: Optional[float] = None,
        min_turn_rate: Optional[float] = None,
        max_turn_rate: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> None:
        self.min_turn_angle = min_turn_angle
        self.max_turn_angle = max_turn_angle
        self.min_turn_rate = min_turn_rate
        self.max_turn_rate = max_turn_rate
        super().__init__(
            event_type=self.event_type,
            termination_mode="ongoing",
            comment=comment,
        )

    def _filter_events(self, event_list: list[dict]) -> list[dict]:
        min_turn_angle, max_turn_angle = check_thresholds(
            self.min_turn_angle, self.max_turn_angle, allow_both_none=True
        )
        min_turn_rate, max_turn_rate = check_thresholds(self.min_turn_rate, self.max_turn_rate, allow_both_none=True)
        valid_events = []
        for e in event_list:
            if (min_turn_angle <= np.abs(e["parameter"]["angle"]) <= max_turn_angle) and (
                ((e["end"] - e["start"]) > 0)
                and min_turn_rate <= np.abs(e["parameter"]["angle"]) / (e["end"] - e["start"]) <= max_turn_rate
            ):
                valid_events.append(e)
        return valid_events
