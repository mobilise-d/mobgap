from typing import Optional

from tpcp import Algorithm
from typing_extensions import Self

from gaitlink.wba._stride_criteria import StrideCriteria


class StrideSelection(Algorithm):
    rules: Optional[list[tuple[str, StrideCriteria]]]

    stride_list: list

    filtered_stride_list_: list
    excluded_stride_list_: list
    exclusion_reasons_: dict[str, tuple[str, StrideCriteria]]

    def __init__(self, rules: Optional[list[tuple[str, StrideCriteria]]]) -> None:
        self.rules = rules

    def filter(self, stride_list: list) -> Self:
        for _, rule in self.rules or []:
            if not isinstance(rule, StrideCriteria):
                raise ValueError("All rules must be instances of `StrideCriteria` or one of its child classes.")

        self.stride_list = sorted(stride_list, key=lambda x: x["start"])
        self.filtered_stride_list_ = []
        self.excluded_stride_list_ = []
        self.exclusion_reasons_ = {}
        for stride in self.stride_list:
            stride_valid, rule = self._check_stride(stride)
            if stride_valid:
                self.filtered_stride_list_.append(stride)
            else:
                self.excluded_stride_list_.append(stride)
                self.exclusion_reasons_[stride["id"]] = rule
        return self

    def _check_stride(self, stride) -> tuple[bool, Optional[tuple[str, StrideCriteria]]]:
        for rule_name, rule in self.rules or []:
            if not rule.check(stride):
                return False, (rule_name, rule)
        return True, None
