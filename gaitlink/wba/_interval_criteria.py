from typing import Optional

import numpy as np
import pandas as pd
from scipy._lib.doccer import inherit_docstring_from
from tpcp import BaseTpcpObject

from gaitlink._docutils import make_filldoc
from gaitlink.wba._utils import check_thresholds


class BaseIntervalCriteria(BaseTpcpObject):
    """Base class for all criteria that are used to filter intervals (e.g. strides) based on their parameters.

    These criteria can be used in combination with the :class:`gaitlink.wba.StrideSelection` algorithm to filter out
    strides that do not meet a set of criteria.
    """

    def check(self, interval: pd.Series) -> bool:
        """Check if the interval meets the criteria.

        Parameters
        ----------
        interval : pd.Series
            The interval to check.
            The interval must at least have a `start` and `end` column.
            Additional columns might be used to check the values of further parameters.

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

    def _get_value(self, interval: pd.Series) -> float:
        raise NotImplementedError("This needs to implemented by child class")

    @inherit_docstring_from(BaseIntervalCriteria)
    def check(self, interval: pd.Series) -> bool:
        value = self._get_value(interval)
        lower_threshold, upper_threshold = check_thresholds(self.lower_threshold, self.upper_threshold)

        # Lower comparison
        operator = np.greater_equal if self.inclusive[0] else np.greater
        lower_comparison = operator(value, lower_threshold)

        # Upper comparison
        operator = np.less_equal if self.inclusive[1] else np.less
        upper_comparison = operator(value, upper_threshold)

        # We convert to bool, so that we don't have to deal with numpy dtypes
        return bool(lower_comparison and upper_comparison)


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

    def _get_value(self, interval: pd.Series) -> float:
        try:
            value = interval[self.parameter]
        except KeyError as e:
            raise ValueError(f"Interval does not contain parameter {self.parameter}") from e
        return value


@_interval_parameter_criteria_docfiller
class IntervalDurationCriteria(_IntervalParameterCriteria):
    """Checks the duration of the stride by subtracting the start and the end value.

    Note that this is different from the `IntervalParameterCriteria` as it does not check a single parameter but
    calculates the duration from the `start` and `end` values.
    In many cases, your interval will have an additional `duration` column that has been calculated beforehand.
    In such cases, you can decide to use the `IntervalParameterCriteria` instead.

    Parameters
    ----------
    %(common_paras)s
    start_col_name
        The name of the column containing the start value. Default: `start`
    end_col_name
        The name of the column containing the end value. Default: `end`

    """

    start_col_name: str
    end_col_name: str

    def __init__(
        self,
        lower_threshold: Optional[float] = None,
        upper_threshold: Optional[float] = None,
        *,
        inclusive: tuple[bool, bool] = (False, True),
        start_col_name: str = "start",
        end_col_name: str = "end",
    ) -> None:
        self.start_col_name = start_col_name
        self.end_col_name = end_col_name
        super().__init__(lower_threshold, upper_threshold, inclusive=inclusive)

    def _get_value(self, interval: pd.Series) -> float:
        try:
            return interval[self.end_col_name] - interval[self.start_col_name]
        except KeyError as e:
            raise ValueError(
                f"Interval does not contain both columns {self.start_col_name} and {self.end_col_name}"
            ) from e
