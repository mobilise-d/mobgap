from typing import ClassVar, Optional

import pandas as pd
from tpcp import Algorithm, cf
from tpcp.misc import set_defaults
from typing_extensions import Self

from mobgap.wba._interval_criteria import (
    BaseIntervalCriteria,
    IntervalDurationCriteria,
    IntervalParameterCriteria,
)


class StrideSelection(Algorithm):
    """Selects strides based on a set of criteria.

    This can be used to filter out valid strides from a list of strides provided by a gait pipeline.

    Parameters
    ----------
    rules
        The rules that are used to filter the strides.
        They need to be provided as a list of tuples, where the first element is the name of the rule and the second
        is an instance of a subclass of :class:`mobgap.wba.BaseIntervalCriteria`.
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
        The dataframe has two columns: ``rule_name`` and ``rule_obj`` corresponding to the values in the `rules`
        parameter.
        The rule always corresponds to the first rule that failed.
        To see the results of all rules, use the ``check_results_`` attribute.
        The df only contains rows for strides that were excluded.
    check_results_
        A dataframe containing the results of the checks for each rule.
        The dataframe has one column for each rule and one row for each stride in the stride list.
        The values are boolean and indicate if the stride meets the criteria of the rule.


    Other Parameters
    ----------------
    stride_list
        The stride list provided to the :meth:`filter` method.
    sampling_rate_hz
        The sampling rate provided to the :meth:`filter` method.

    """

    _action_methods = ("filter",)
    _composite_params = ("rules",)

    rules: Optional[list[tuple[str, BaseIntervalCriteria]]]

    stride_list: pd.DataFrame
    sampling_rate_hz: float

    _exclusion_reasons: pd.DataFrame
    check_results_: pd.DataFrame

    class PredefinedParameters:
        mobilise_stride_selection: ClassVar = {
            "rules": [
                (
                    "stride_duration_thres",
                    IntervalDurationCriteria(min_duration_s=0.2, max_duration_s=3.0),
                ),
                (
                    "stride_length_thres",
                    IntervalParameterCriteria("stride_length", lower_threshold=0.15, upper_threshold=None),
                ),
            ],
        }

    @set_defaults(**{k: cf(v) for k, v in PredefinedParameters.mobilise_stride_selection.items()})
    def __init__(self, rules: Optional[list[tuple[str, BaseIntervalCriteria]]]) -> None:
        self.rules = rules

    @property
    def filtered_stride_list_(self) -> pd.DataFrame:
        return self.stride_list[self._exclusion_reasons["rule_name"].isna()].copy()

    @property
    def excluded_stride_list_(self) -> pd.DataFrame:
        return self.stride_list[self._exclusion_reasons["rule_name"].notna()].copy()

    @property
    def exclusion_reasons_(self) -> pd.DataFrame:
        return self._exclusion_reasons[self._exclusion_reasons["rule_name"].notna()]

    def filter(self, stride_list: pd.DataFrame, *, sampling_rate_hz: float) -> Self:
        """Filter the stride list.

        Parameters
        ----------
        stride_list
            The stride list to filter.
            The stride list must be a dataframe, where each row represents a stride.
            The index should be a unique identifier for each stride.
            The dataframe must at least have a `start` and `end` column.
            We assume that the `start` and `end` values are in samples and can be converted to seconds using the
            `sampling_rate_hz` parameter.
            Additional required columns depend on the rules that are used for filtering.
        sampling_rate_hz
            The sampling rate of the data in Hz.
            This is used to potentially convert the ``start`` and ``end`` values of a stride to seconds, assuming that
            they are in samples.
            If this is not the case and the value are already in seconds, `sampling_rate_hz` should set to 1.

        Returns
        -------
        self
            The instance itself with the result parameters set.
        """
        # TODO: Add better checking for compound fields dtype
        for _, rule in self.rules or []:
            if not isinstance(rule, BaseIntervalCriteria):
                raise TypeError("All rules must be instances of `IntervalSummaryCriteria` or one of its child classes.")

        self.stride_list = stride_list
        self.sampling_rate_hz = sampling_rate_hz

        if self.rules is None:
            self._exclusion_reasons = pd.DataFrame(columns=["rule_name", "rule_obj"]).reindex(stride_list.index)
            return self

        rules_as_dict = dict(self.rules)

        self.check_results_ = pd.concat(
            {
                name: rule.check_multiple(stride_list, sampling_rate_hz=sampling_rate_hz)
                for name, rule in rules_as_dict.items()
            },
            axis=1,
        )

        # find first rule that fails:
        # idxmin will return the first False, but will always return a rule, even if everything is True
        self._exclusion_reasons = (
            self.check_results_.idxmin(axis=1)
            # So we need to take a second step where we replace all cases where all are True with Nan
            .where(~self.check_results_.all(axis=1))
            # Then we rename
            .rename("rule_name")
            .to_frame()
            # And find the corresponding rule object
            .assign(rule_obj=lambda df_: df_["rule_name"].replace(rules_as_dict))
        )

        return self
