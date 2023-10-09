from typing import Optional

from tpcp import BaseTpcpObject

from gaitlink.wba._utils import check_thresholds


class StrideCriteria(BaseTpcpObject):
    comment: Optional[str]

    def __init__(self, comment: Optional[str] = None) -> None:
        self.comment = comment

    def check(self, stride: dict) -> bool:
        raise NotImplementedError("This needs to implemented by child class")

    def check_stride_list(self, stride_list: list[dict]) -> bool:
        raise NotImplementedError("This needs to implemented by child class")


class ThresholdCriteria(StrideCriteria):
    lower_threshold: float
    upper_threshold: float
    parameter: str

    def __init__(
        self,
        parameter: str,
        lower_threshold: Optional[float] = None,
        upper_threshold: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> None:
        lower_threshold, upper_threshold = check_thresholds(lower_threshold, upper_threshold)

        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.parameter = parameter

        super().__init__(comment=comment)

    def check(self, stride: dict) -> bool:
        return self.lower_threshold < stride["parameter"][self.parameter] <= self.upper_threshold

    def check_stride_list(self, stride_list: list[dict]) -> bool:
        return all(self.check(s) is not False for s in stride_list)


class StrideTimeCriteria(StrideCriteria):
    """Checks the duration of the stride by subtracting the start and the end timestamp."""

    lower_threshold: float
    upper_threshold: float

    def __init__(
        self,
        lower_threshold: Optional[float] = None,
        upper_threshold: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> None:
        lower_threshold, upper_threshold = check_thresholds(lower_threshold, upper_threshold)

        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

        super().__init__(comment=comment)

    def check(self, stride: dict) -> bool:
        # TODO: Test
        stride_time = stride["end"] - stride["start"]
        return self.lower_threshold < stride_time <= self.upper_threshold

    def check_stride_list(self, stride_list: list[dict]) -> bool:
        return all(self.check(s) is not False for s in stride_list)
