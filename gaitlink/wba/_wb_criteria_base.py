from typing import Optional

import numpy as np
from tpcp import BaseTpcpObject
from typing_extensions import Literal

from gaitlink.wba._utils import check_thresholds


class WBCriteria(BaseTpcpObject):
    comment: Optional[str]

    def __init__(self, comment: Optional[str] = None) -> None:
        self.comment = comment

    def check_wb_start_end(
        self,
        stride_list: list[dict],
        original_start: int,
        current_start: int,
        current_end: int,
        event_list: Optional[list[dict]] = None,
    ) -> tuple[Optional[int], Optional[int]]:
        """Determine the current start and end of the current WB.

        This method gets passed all strides (past and future) of the entire measurement.
        All this information can be used in the rule.
        However, looking in the future is discouraged, as this might interfere with other rules.

        Parameters
        ----------
        stride_list
            A list of all strides within the measurement.
        event_list
            A nested list of all events of the measurement
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

    def check_include(self, preliminary_wb: dict, event_list: Optional[list[dict]] = None) -> bool:
        """Check if a preliminary WB should be considered an actual WB.

        Parameters
        ----------
        preliminary_wb
            The preliminary wb including its stride list
        event_list
            A nested list of all events of the measurement

        Returns
        -------
        is_actual_wb
            True or False if the preliminary WB should be considered an actual WB
        """
        return True


class SummaryCriteria(WBCriteria):
    lower_threshold: float
    upper_threshold: float

    def __init__(
        self,
        lower_threshold: Optional[float] = None,
        upper_threshold: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> None:
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

        super().__init__(comment=comment)

    def check_include(self, wb: dict, event_list: Optional[list[dict]] = None) -> bool:
        lower_threshold, upper_threshold = check_thresholds(self.lower_threshold, self.upper_threshold)
        stride_list = wb["strideList"]
        value = self._calc_summary(stride_list)
        return lower_threshold <= value <= upper_threshold

    def _calc_summary(self, stride_list: list[dict]) -> float:
        raise NotImplementedError("This needs to implemented by child class")


class EventCriteria:
    event_type: str

    def filter_events(self, event_list):
        if event_list is None:
            raise ValueError("You are using an event based Criteria, without providing any events.")
        events = next(
            (event["events"] for event in event_list if event["name"] == self.event_type),
            [],
        )
        events = self._filter_events(events)
        return events

    @staticmethod
    def _convert_to_start_stop(events):
        event_start_end = np.array([[event["start"], event["end"]] for event in events])
        event_start_end = event_start_end[event_start_end[:, 0].argsort()]
        return event_start_end

    def _filter_events(self, event_list: list[dict]) -> list[dict]:
        return event_list


# TODO: "Precompile relevant event list for performance
class EventTerminationCriteria(WBCriteria, EventCriteria):
    """Terminate/prevent starts of WBs in case strides overlap with events.

    Parameters
    ----------
    event_type
        Which type of event (from the event list) should be used
    termination_mode
        This controls under which circumstances a stride should be considered invalid for the WB.

        "start": A stride is invalid, if since the beginning of the last stride a new event was started.
        "end": A stride is invalid, if an event was ended since the beginning of the last stride
        "both": A stride is invalid, if any of the above conditions applied
        "ongoing": A stride is invalid, if it has any overlap with any event.

        At the very beginning of a recording, the start of the last stride is equal to the start of the recording.

    """

    termination_mode: str

    _termination_modes = (
        "start",
        "end",
        "both",
        "ongoing",
    )

    def __init__(
        self,
        event_type: str,
        termination_mode: Literal["start", "end", "both", "ongoing"] = "ongoing",
        comment: Optional[str] = None,
    ) -> None:
        self.event_type = event_type
        self.termination_mode = termination_mode
        super().__init__(comment=comment)

    def check_wb_start_end(
        self,
        stride_list: list[dict],
        original_start: int,
        current_start: int,
        current_end: int,
        event_list: Optional[list[dict]] = None,
    ) -> tuple[Optional[int], Optional[int]]:
        last_stride_start = 0
        if current_end >= 1:
            last_stride_start = stride_list[current_end - 1]["start"]
        current_stride_end = stride_list[current_end]["end"]
        current_stride_valid = self._check_stride(last_stride_start, current_stride_end, event_list)
        if current_stride_valid:
            return None, None
        # If the current stride is not valid, we either need to delay the start, if no proper WB has been started yet,
        # or we need to terminate the WB.
        if current_end == current_start:
            # Prevent the start
            return current_end + 1, None
        # Terminate the WB
        return None, current_end - 1

    def _check_stride(self, last_stride_start: float, current_stride_end: float, event_list):
        if self.termination_mode not in self._termination_modes:
            # We do this check here to avoid computing the events if the termination mode is invalid
            raise ValueError(f'"termination_mode" must be one of {self._termination_modes}')
        events = self.filter_events(event_list)
        if not events:
            return True
        event_start_end = self._convert_to_start_stop(events)

        events_started_since_last_stride = event_start_end[
            np.nonzero((event_start_end[:, 0] >= last_stride_start) & (event_start_end[:, 0] < current_stride_end))
        ]

        events_ended_since_last_stride = event_start_end[
            np.nonzero((event_start_end[:, 1] >= last_stride_start) & (event_start_end[:, 1] < current_stride_end))
        ]

        if self.termination_mode == "start":
            return len(events_started_since_last_stride) == 0
        if self.termination_mode == "end":
            return len(events_ended_since_last_stride) == 0
        if self.termination_mode == "both":
            return len(events_started_since_last_stride) == 0 and len(events_ended_since_last_stride) == 0
        if self.termination_mode == "ongoing":
            # Find events that where started before and are still ongoing
            ongoing_events = event_start_end[
                np.nonzero((event_start_end[:, 0] <= last_stride_start) & (event_start_end[:, 1] >= current_stride_end))
            ]
            return len(events_started_since_last_stride) == 0 and len(ongoing_events) == 0
        # We never reach this point
        raise ValueError()


class EventInclusionCriteria(WBCriteria, EventCriteria):
    """Test if a WB is fully or partially covered by an event."""

    event_type: str
    overlap: str

    _overlap_types = ("partial", "contains", "is_contained", "no_overlap")

    def __init__(
        self,
        event_type: str,
        overlap: str = "partial",
        comment: Optional[str] = None,
    ) -> None:
        self.overlap = overlap
        self.event_type = event_type

        super().__init__(comment=comment)

    def check_include(self, wb: dict, event_list: Optional[list[dict]] = None) -> bool:
        # TODO: TEST
        events = self.filter_events(event_list)
        if not events:
            return True
        event_start_end = self._convert_to_start_stop(events)

        min_ends = np.minimum(event_start_end[:, 1], wb["end"])
        max_start = np.maximum(event_start_end[:, 0], wb["start"])
        amount_overlap = min_ends - max_start

        if self.overlap == "contains":
            return len(event_start_end[amount_overlap >= event_start_end[:, 1] - event_start_end[:, 0]]) > 0
        if self.overlap == "is_contained":
            return len(event_start_end[amount_overlap >= wb["end"] - wb["start"]]) > 0
        if self.overlap == "no_overlap":
            return len(event_start_end[amount_overlap > 0]) == 0
        if self.overlap == "partial":
            return len(event_start_end[amount_overlap > 0]) > 0
        raise ValueError(f'"overlap" must be one of {self._overlap_types}')


class EndOfList(WBCriteria):
    """Dummy criteria to describe the end of the stride list.

    DO NOT USE THIS AS A CUSTOM RULE
    """
