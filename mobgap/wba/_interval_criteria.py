from typing import Optional

import pandas as pd
from scipy._lib.doccer import inherit_docstring_from
from tpcp import BaseTpcpObject

from mobgap._docutils import make_filldoc
from mobgap.wba._utils import compare_with_threshold_multiple


class BaseIntervalCriteria(BaseTpcpObject):
    """Base class for all criteria that are used to filter intervals (e.g. strides) based on their parameters.

    These criteria can be used in combination with the :class:`mobgap.wba.StrideSelection` algorithm to filter out
    strides that do not meet a set of criteria.
    """

    def check(self, interval: pd.Series, *, sampling_rate_hz: Optional[float] = None) -> bool:
        """Check if the interval meets the criteria.

        Parameters
        ----------
        interval : pd.Series
            The interval to check.
            The interval must at least have a `start` and `end` column that contain the start and end of the interval
            in samples.
            Additional columns might be used to check the values of further parameters.
        sampling_rate_hz
            The sampling rate of the data in Hz.
            This is used to potentially convert the `start` and `end` values to seconds, assuming that they are in
            samples.
            If this is not the case and the value are already in seconds, `sampling_rate_hz` should set to 1.

        """
        return bool(self.check_multiple(pd.DataFrame([interval]), sampling_rate_hz=sampling_rate_hz).iloc[0])

    def requires_columns(self) -> list[str]:
        """Return a list of columns that are required in the intervals to check.

        Returns
        -------
        list[str]
            A list of column names that are required in the intervals to check.

        """
        raise NotImplementedError("This needs to implemented by child class")

    def check_multiple(self, intervals: pd.DataFrame, *, sampling_rate_hz: Optional[float] = None) -> pd.Series:
        """Check if the intervals meet the criteria.

        Parameters
        ----------
        intervals : pd.DataFrame
            The intervals to check.
            The intervals must at least have a `start` and `end` column that contain the start and end of the interval
            in samples.
            Additional columns might be used to check the values of further parameters.
        sampling_rate_hz
            The sampling rate of the data in Hz.
            This is used to potentially convert the `start` and `end` values to seconds, assuming that they are in
            samples.
            If this is not the case and the value are already in seconds, `sampling_rate_hz` should set to 1.

        Returns
        -------
        pd.Series
            A boolean series indicating if the intervals meet the criteria.

        """
        raise NotImplementedError("This needs to implemented by child class")


_interval_parameter_criteria_docfiller = make_filldoc(
    {
        "common_paras": """
    lower_threshold
        The lower threshold for the parameter.
        If `None`, the lower threshold is not checked.
    upper_threshold
        The upper threshold for the parameter.
        If `None`, the upper threshold is not checked.
    inclusive
        A tuple of two booleans indicating if the lower and upper threshold should be inclusive in the comparison.
        By default, the lower threshold is exclusive and the upper threshold is inclusive.
    """
    },
)


class _IntervalParameterCriteria(BaseIntervalCriteria):
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

    def _get_values(self, intervals: pd.DataFrame, sampling_rate_hz: Optional[float]) -> pd.Series:
        raise NotImplementedError("This needs to implemented by child class")

    @inherit_docstring_from(BaseIntervalCriteria)
    def check_multiple(self, intervals: pd.DataFrame, *, sampling_rate_hz: Optional[float] = None) -> pd.Series:
        values = self._get_values(intervals, sampling_rate_hz)
        return compare_with_threshold_multiple(values, self.lower_threshold, self.upper_threshold, self.inclusive)


@_interval_parameter_criteria_docfiller
class IntervalParameterCriteria(_IntervalParameterCriteria):
    """Checks if a parameter of the interval meets a threshold.

    Parameters
    ----------
    parameter
        The name of the parameter to check.
    %(common_paras)s

    """

    parameter: str

    def __init__(
        self,
        parameter: str,
        lower_threshold: Optional[float] = None,
        upper_threshold: Optional[float] = None,
        *,
        inclusive: tuple[bool, bool] = (False, True),
    ) -> None:
        self.parameter = parameter
        super().__init__(lower_threshold, upper_threshold, inclusive=inclusive)

    def requires_columns(self) -> list[str]:
        return [self.parameter]

    def _get_values(self, intervals: pd.DataFrame, sampling_rate_hz: Optional[float]) -> pd.Series:  # noqa: ARG002
        try:
            values = intervals[self.parameter]
        except KeyError as e:
            raise ValueError(f"Intervals do not contain parameter {self.parameter}") from e
        return values


@_interval_parameter_criteria_docfiller
class IntervalDurationCriteria(BaseIntervalCriteria):
    """Checks the duration of the stride by subtracting the start and the end value.

    Note that this is different from the `IntervalParameterCriteria` as it does not check a single parameter but
    calculates the duration from the `start` and `end` values.
    In many cases, your interval will have an additional `duration` column that has been calculated beforehand.
    In such cases, you can decide to use the `IntervalParameterCriteria` instead.

    Parameters
    ----------
    %(common_paras)s

    """

    _START_COL_NAME: str = "start"
    _END_COL_NAME: str = "end"

    min_duration_s: float
    max_duration_s: float
    inclusive: tuple[bool, bool]

    def __init__(
        self,
        min_duration_s: Optional[float] = None,
        max_duration_s: Optional[float] = None,
        *,
        inclusive: tuple[bool, bool] = (False, True),
    ) -> None:
        self.min_duration_s = min_duration_s
        self.max_duration_s = max_duration_s
        self.inclusive = inclusive

    def requires_columns(self) -> list[str]:
        return [self._START_COL_NAME, self._END_COL_NAME]

    def _get_values(
        self,
        intervals: pd.DataFrame,
        sampling_rate_hz: Optional[float],
    ) -> pd.Series:
        if sampling_rate_hz is None:
            raise ValueError("The sampling rate must be provided if the IntervalDurationCriteria is used.")
        try:
            return (intervals[self._END_COL_NAME] - intervals[self._START_COL_NAME]) / sampling_rate_hz
        except KeyError as e:
            raise ValueError(
                f"Interval does not contain both columns {self._START_COL_NAME} and {self._END_COL_NAME}"
            ) from e

    @inherit_docstring_from(BaseIntervalCriteria)
    def check_multiple(self, intervals: pd.DataFrame, *, sampling_rate_hz: Optional[float] = None) -> pd.Series:
        values = self._get_values(intervals, sampling_rate_hz)
        return compare_with_threshold_multiple(values, self.min_duration_s, self.max_duration_s, self.inclusive)
