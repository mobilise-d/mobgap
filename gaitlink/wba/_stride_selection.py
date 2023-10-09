from typing import Optional

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self

from gaitlink.wba._interval_criteria import IntervalSummaryCriteria


class StrideSelection(Algorithm):
    rules: Optional[list[tuple[str, IntervalSummaryCriteria]]]

    stride_list: pd.DataFrame

    exclusion_reasons_: pd.DataFrame

    def __init__(self, rules: Optional[list[tuple[str, IntervalSummaryCriteria]]]) -> None:
        self.rules = rules

    @property
    def filtered_stride_list_(self) -> pd.DataFrame:
        return self.stride_list[self.exclusion_reasons_["rule_name"].isna()]

    @property
    def excluded_stride_list_(self) -> pd.DataFrame:
        return self.stride_list[self.exclusion_reasons_["rule_name"].notna()]

    def filter(self, stride_list: pd.DataFrame) -> Self:
        for _, rule in self.rules or []:
            if not isinstance(rule, IntervalSummaryCriteria):
                raise ValueError(
                    "All rules must be instances of `IntervalSummaryCriteria` or one of its child classes."
                )

        self.stride_list = stride_list
        self.exclusion_reasons_ = self.stride_list.sort_values("start").apply(
            lambda x: pd.Series(self._check_stride(x), index=["rule_name", "rule_obj"]), axis=1
        )

        return self

    def _check_stride(self, stride) -> tuple[Optional[str], Optional[IntervalSummaryCriteria]]:
        for rule_name, rule in self.rules or []:
            if not rule.check(stride):
                return rule_name, rule
        return None, None
