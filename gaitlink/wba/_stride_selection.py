from typing import Optional

import pandas as pd
from tpcp import Algorithm, cf
from typing_extensions import Self

from gaitlink.wba._interval_criteria import BaseIntervalCriteria, IntervalDurationCriteria, IntervalParameterCriteria

# : The default rules that are used for stride selection in the Mobilise-D pipeline.
default_mobilised_stride_selection_rules = [
    ("stride_duration_thres", IntervalDurationCriteria(lower_threshold=0.2, upper_threshold=3.0)),
    ("stride_length_thres", IntervalParameterCriteria("stride_length", lower_threshold=0.15, upper_threshold=None)),
]


class StrideSelection(Algorithm):
    """Selects strides based on a set of criteria.

    This can be used to filter out valid strides from a list of strides provided by a gait pipeline.

    Parameters
    ----------
    rules
        The rules that are used to filter the strides.
        They need to be provided as a list of tuples, where the first element is the name of the rule and the second
        is an instance of a subclass of :class:`gaitlink.wba.BaseIntervalCriteria`.
        If `None`, no rules are applied and all strides are considered valid.

    Attributes
    ----------
    filtered_stride_list_
        A dataframe containing all strides that are considered valid.
        This is a strict subset of the stride list that was provided to the :meth:`filter` method.
    excluded_stride_list_
        A dataframe containing all strides that are considered invalid.
        This is a strict subset of the stride list that was provided to the :meth:`filter` method.
    exclusion_reasons_
        A dataframe containing the reason why a stride was excluded.
        The dataframe has two columns: `rule_name` and `rule_obj` corresponding to the values in the `rules` parameter.
        The df only contains rows for strides that were excluded.


    Other Parameters
    ----------------
    stride_list
        The stride list provided to the :meth:`filter` method.

    """

    _action_methods = ("filter",)
    _composite_params = ("rules",)

    rules: Optional[list[tuple[str, BaseIntervalCriteria]]]

    stride_list: pd.DataFrame

    _exclusion_reasons: pd.DataFrame

    def __init__(
        self, rules: Optional[list[tuple[str, BaseIntervalCriteria]]] = cf(default_mobilised_stride_selection_rules)
    ) -> None:
        self.rules = rules

    @property
    def filtered_stride_list_(self) -> pd.DataFrame:
        return self.stride_list[self._exclusion_reasons["rule_name"].isna()]

    @property
    def excluded_stride_list_(self) -> pd.DataFrame:
        return self.stride_list[self._exclusion_reasons["rule_name"].notna()]

    @property
    def exclusion_reasons_(self) -> pd.DataFrame:
        return self._exclusion_reasons[self._exclusion_reasons["rule_name"].notna()]

    def filter(self, stride_list: pd.DataFrame) -> Self:  # noqa: A003
        """Filter the stride list.

        Parameters
        ----------
        stride_list
            The stride list to filter.
            The stride list must be a dataframe, where each row represents a stride.
            The index should be a unique identifier for each stride.
            The dataframe must at least have a `start` and `end` column.
            Additional required columns depend on the rules that are used for filtering.

        """
        # TODO: Add better checking for compound fields dtype
        for _, rule in self.rules or []:
            if not isinstance(rule, BaseIntervalCriteria):
                raise TypeError("All rules must be instances of `IntervalSummaryCriteria` or one of its child classes.")

        self.stride_list = stride_list
        self._exclusion_reasons = self.stride_list.apply(
            lambda x: pd.Series(self._check_stride(x), index=["rule_name", "rule_obj"]), axis=1
        )

        return self

    def _check_stride(self, stride: pd.Series) -> tuple[Optional[str], Optional[BaseIntervalCriteria]]:
        for rule_name, rule in self.rules or []:
            if not rule.check(stride):
                return rule_name, rule
        return None, None
