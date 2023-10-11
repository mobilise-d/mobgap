import uuid
from typing import Callable, Optional

import numpy as np
from tpcp import Algorithm
from typing_extensions import Self

from gaitlink.wba._wb_criteria_base import BaseWBCriteria, EndOfList


# TODO: Update namings
# TODO: Change default outputs to dataframes?
def create_wb(stride_list: list[dict]) -> dict:
    wb = {
        "id": str(uuid.uuid1()),
        "nStrides": len(stride_list),
        "start": stride_list[0]["start"],
        "end": stride_list[-1]["end"],
        "timeUnit": stride_list[0].get("timeUnit", None),
        "strideList": stride_list,
    }
    return wb


class WBAssembly(Algorithm):
    _action_methods = ("assemble",)
    _composite_params = ("rules",)

    post_processors: Optional[list[Callable]]
    rules: Optional[list[tuple[str, BaseWBCriteria]]]

    stride_list: list[dict]
    event_list: list[dict[str, list[dict]]]

    wb_list_: list[dict]
    excluded_wb_list_: list[dict]
    termination_reasons_: dict[str, tuple[str, BaseWBCriteria]]
    exclusion_reasons_: dict[str, tuple[str, BaseWBCriteria]]

    _excluded_strides_: list[dict]
    _stride_exclusion_reasons_: dict[str, tuple[str, BaseWBCriteria]]

    def __init__(self, rules: Optional[list[tuple[str, BaseWBCriteria]]] = None) -> None:
        self.rules = rules
        # rules = rules or []

        # rules = rules + list(self._default_rules)
        # self.rules = rules

    @property
    def filtered_stride_list_(self) -> list[dict]:
        all_strides = []
        for wb in self.wb_list_:
            all_strides.extend(wb["strideList"])
        return all_strides

    @property
    def excluded_stride_list_(self) -> list[dict]:
        all_strides = self._wb_excluded_stride_list_
        all_strides.extend(self._excluded_strides_)
        return all_strides

    @property
    def _wb_excluded_stride_list_(self) -> list[dict]:
        all_strides = []
        for wb in self.excluded_wb_list_:
            all_strides.extend(wb["strideList"])
        return all_strides

    @property
    def stride_exclusion_reasons_(self) -> dict[str, BaseWBCriteria]:
        all_reasons = {}
        for wb in self.excluded_wb_list_:
            all_reasons = {**all_reasons, **{s["id"]: self.exclusion_reasons_[wb["id"]] for s in wb["strideList"]}}
        all_reasons = {**all_reasons, **self._stride_exclusion_reasons_}
        return all_reasons

    def assemble(
        self,
        stride_list: list,
        event_list: Optional[list[dict[str, list[dict]]]] = None,
    ) -> Self:
        self.stride_list = stride_list
        stride_list_sorted = sorted(self.stride_list, key=lambda x: x["start"])
        self.event_list = event_list
        for _rule_name, rule in self.rules or []:
            if not isinstance(rule, BaseWBCriteria):
                raise ValueError("All rules must be instances of `WBCriteria` or one of its child classes.")

        (
            preliminary_wb_list,
            excluded_wb_list,
            exclusion_reasons,
            termination_reasons,
            excluded_strides,
            stride_exclusion_reasons,
        ) = self._apply_termination_rules(stride_list_sorted, self.event_list)
        self.wb_list_, excluded_wb_list_2, exclusion_reasons_2 = self._apply_inclusion_rules(
            preliminary_wb_list, event_list
        )
        self.termination_reasons_ = termination_reasons
        self.excluded_wb_list_ = [*excluded_wb_list, *excluded_wb_list_2]
        self.exclusion_reasons_ = {**exclusion_reasons, **exclusion_reasons_2}
        self._excluded_strides_ = excluded_strides
        self._stride_exclusion_reasons_ = stride_exclusion_reasons
        return self

    def _apply_termination_rules(
        self, stride_list, event_list
    ) -> tuple[
        list[dict],
        list[dict],
        dict[str, tuple[str, BaseWBCriteria]],
        dict[str, tuple[str, BaseWBCriteria]],
        list[dict],
        dict[str, tuple[str, BaseWBCriteria]],
    ]:
        end = 0
        preliminary_wb_list = []
        termination_reasons = {}
        excluded_wb_list = []
        exclusion_reasons = {}
        excluded_strides = []
        stride_exclusion_reasons = {}
        while end < len(stride_list):
            start = end
            final_start, final_end, start_delay_reason, termination_reason = self._find_first_preliminary_wb(
                stride_list, start, event_list
            )
            if final_start >= final_end:
                # There was a termination criteria, but the WB was never properly started
                final_start = final_end + 1
                removed_wb = create_wb(stride_list[start:final_start])
                excluded_wb_list.append(removed_wb)
                exclusion_reasons[removed_wb["id"]] = start_delay_reason
                termination_reasons[removed_wb["id"]] = termination_reason
            else:
                # The preliminary WB is saved
                preliminary_wb = create_wb(stride_list[final_start : final_end + 1])
                preliminary_wb_list.append(preliminary_wb)
                termination_reasons[preliminary_wb["id"]] = termination_reason
                # Save strides that were excluded in the beginning as a excluded strides
                removed_strides = stride_list[start:final_start]
                excluded_strides.extend(removed_strides)
                # All strides that were considered as invalid start strides will be removed.
                for s in removed_strides:
                    stride_exclusion_reasons[s["id"]] = start_delay_reason

            end = final_end + 1
        return (
            preliminary_wb_list,
            excluded_wb_list,
            exclusion_reasons,
            termination_reasons,
            excluded_strides,
            stride_exclusion_reasons,
        )

    def _find_first_preliminary_wb(
        self,
        stride_list: list[dict],
        original_start,
        event_list: Optional[list[dict]] = None,
    ) -> tuple[int, int, Optional[tuple[str, BaseWBCriteria]], Optional[tuple[str, BaseWBCriteria]]]:
        # TODO: Filter evenetlist here to not have any events that are before the window anyway.
        end_index = len(stride_list)
        current_end = original_start
        current_start = original_start
        start_delay_reason = None
        while current_end < end_index:
            tmp_start, tmp_end, tmp_start_delay_reason, termination_reason = self._check_wb_start_end(
                stride_list, original_start, current_start, current_end, event_list
            )
            if termination_reason:
                # In case of termination return directly.
                # Do not consider further updates on the start as they might be made based on strides that are not part
                # of a WB
                return current_start, tmp_end, start_delay_reason, termination_reason
            if tmp_start_delay_reason:
                start_delay_reason = tmp_start_delay_reason
                current_start = tmp_start

            current_end += 1
        # In case we end the loop without any termination rule firing
        return current_start, len(stride_list), start_delay_reason, ("end_of_list", EndOfList())

    def _check_wb_start_end(
        self,
        stride_list: list[dict],
        original_start,
        current_start,
        current_end,
        event_list: Optional[list[dict]] = None,
    ) -> tuple[int, int, Optional[tuple[str, BaseWBCriteria]], Optional[tuple[str, BaseWBCriteria]]]:
        termination_rule = None
        start_delay_rule = None
        tmp_start = -1
        tmp_end = np.inf
        for rule_name, rule in self.rules or []:
            start, end = rule.check_wb_start_end(stride_list, original_start, current_start, current_end, event_list)
            if start is not None and start > tmp_start:
                tmp_start = start
                start_delay_rule = (rule_name, rule)
            if end is not None and end < tmp_end:
                tmp_end = end
                termination_rule = (rule_name, rule)
        return tmp_start, tmp_end, start_delay_rule, termination_rule

    def _apply_inclusion_rules(
        self, preliminary_wb_list: list[dict], event_list
    ) -> tuple[list[dict], list[dict], dict[str, tuple[str, BaseWBCriteria]]]:
        wb_list = []
        removed_wb_list_ = []
        exclusion_reasons = {}
        for wb in preliminary_wb_list:
            for rule_name, rule in self.rules or []:
                if not rule.check_include(wb, event_list):
                    removed_wb_list_.append(wb)
                    exclusion_reasons[wb["id"]] = (rule_name, rule)
                    break
            else:
                wb_list.append(wb)
        return wb_list, removed_wb_list_, exclusion_reasons
