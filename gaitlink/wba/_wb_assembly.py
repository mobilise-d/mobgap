import uuid
from typing import Optional

import numpy as np
import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self

from gaitlink.wba._wb_criteria_base import BaseWBCriteria, EndOfList


class WBAssembly(Algorithm):
    _action_methods = ("assemble",)
    _composite_params = ("rules",)

    rules: Optional[list[tuple[str, BaseWBCriteria]]]

    stride_list: pd.DataFrame

    annotated_stride_list_: pd.DataFrame
    excluded_stride_list_: pd.DataFrame
    termination_reasons_: dict[str, tuple[str, BaseWBCriteria]]
    exclusion_reasons_: dict[str, tuple[str, BaseWBCriteria]]

    def __init__(self, rules: Optional[list[tuple[str, BaseWBCriteria]]] = None) -> None:
        self.rules = rules

    @property
    def wbs_(self) -> dict[str, pd.DataFrame]:
        return {k: v.drop("wb_id", axis=1) for k, v in self.annotated_stride_list_.groupby("wb_id")}

    @property
    def excluded_wbs_(self) -> dict[str, pd.DataFrame]:
        return {
            k: v.drop("wb_id", axis=1)
            # We group only the strides that are part of a preliminary WB
            for k, v in self.excluded_stride_list_[self.excluded_stride_list_["pre_wb_id"].notna()].groupby("pre_wb_id")
        }

    def assemble(
        self,
        stride_list: pd.DataFrame,
    ) -> Self:
        # TODO: Add better checks for correct type of compound rule field
        self.stride_list = stride_list
        stride_list_sorted = self.stride_list.sort_values(by=["start", "end"])
        for _, rule in self.rules or []:
            if not isinstance(rule, BaseWBCriteria):
                raise TypeError("All rules must be instances of `WBCriteria` or one of its child classes.")

        (
            preliminary_wb_list,
            excluded_wb_list,
            exclusion_reasons,
            termination_reasons,
            excluded_strides,
            stride_exclusion_reasons,
        ) = self._apply_termination_rules(stride_list_sorted)
        wb_list, excluded_wb_list_2, exclusion_reasons_2 = self._apply_inclusion_rules(preliminary_wb_list)
        self.annotated_stride_list_ = pd.concat(wb_list, names=["wb_id", "s_id"]).reset_index("wb_id")

        if len(combined_excluded_stride_list := {**excluded_wb_list, **excluded_wb_list_2}) > 0:
            excluded_strides_in_wbs = pd.concat(combined_excluded_stride_list, names=["pre_wb_id", "s_id"]).reset_index(
                "pre_wb_id"
            )
        else:
            excluded_strides_in_wbs = pd.DataFrame(columns=stride_list.columns)
        other_excluded_strides = excluded_strides.assign(pre_wb_id=None)
        self.excluded_stride_list_ = pd.concat([excluded_strides_in_wbs, other_excluded_strides])
        self.termination_reasons_ = termination_reasons
        self.exclusion_reasons_ = {**exclusion_reasons, **exclusion_reasons_2}
        return self

    def _apply_termination_rules(
        self, stride_list: pd.DataFrame
    ) -> tuple[
        dict[str, pd.DataFrame],
        dict[str, pd.DataFrame],
        dict[str, tuple[str, BaseWBCriteria]],
        dict[str, tuple[str, BaseWBCriteria]],
        pd.DataFrame,
        dict[str, tuple[str, BaseWBCriteria]],
    ]:
        end = 0
        preliminary_wb_list = {}
        termination_reasons = {}
        excluded_wb_list = {}
        exclusion_reasons = {}
        excluded_strides = []
        stride_exclusion_reasons = {}
        while end < len(stride_list):
            start = end
            final_start, final_end, start_delay_reason, termination_reason = self._find_first_preliminary_wb(
                stride_list, start
            )
            preliminary_wb_id = str(uuid.uuid1())

            if final_start >= final_end:
                # There was a termination criteria, but the WB was never properly started
                final_start = final_end + 1
                excluded_wb_list[preliminary_wb_id] = stride_list.iloc[start:final_start]
                exclusion_reasons[preliminary_wb_id] = start_delay_reason
                termination_reasons[preliminary_wb_id] = termination_reason
            else:
                # The preliminary WB is saved
                preliminary_wb_list[preliminary_wb_id] = stride_list.iloc[final_start : final_end + 1]
                termination_reasons[preliminary_wb_id] = termination_reason
                # Save strides that were excluded in the beginning as an excluded strides
                removed_strides = stride_list.iloc[start:final_start]
                if len(removed_strides) > 0:
                    excluded_strides.append(removed_strides)
                # All strides that were considered as invalid start strides will be removed.
                for s in removed_strides.index:
                    stride_exclusion_reasons[s] = start_delay_reason

            end = final_end + 1
        if len(excluded_strides) > 0:
            excluded_strides = pd.concat(excluded_strides)
        else:
            excluded_strides = pd.DataFrame(columns=stride_list.columns)
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
        stride_list: pd.DataFrame,
        original_start: int,
    ) -> tuple[int, int, Optional[tuple[str, BaseWBCriteria]], Optional[tuple[str, BaseWBCriteria]]]:
        end_index = len(stride_list)
        current_end = original_start
        current_start = original_start
        start_delay_reason = None
        while current_end < end_index:
            tmp_start, tmp_end, tmp_start_delay_reason, termination_reason = self._check_wb_start_end(
                stride_list, original_start, current_start, current_end
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
        stride_list: pd.DataFrame,
        original_start: int,
        current_start: int,
        current_end: int,
    ) -> tuple[int, int, Optional[tuple[str, BaseWBCriteria]], Optional[tuple[str, BaseWBCriteria]]]:
        termination_rule = None
        start_delay_rule = None
        tmp_start = -1
        tmp_end = np.inf
        for rule_name, rule in self.rules or []:
            start, end = rule.check_wb_start_end(
                stride_list, original_start=original_start, current_start=current_start, current_end=current_end
            )
            if start is not None and start > tmp_start:
                tmp_start = start
                start_delay_rule = (rule_name, rule)
            if end is not None and end < tmp_end:
                tmp_end = end
                termination_rule = (rule_name, rule)
        return tmp_start, tmp_end, start_delay_rule, termination_rule

    def _apply_inclusion_rules(
        self, preliminary_wb_list: dict[str, pd.DataFrame]
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, tuple[str, BaseWBCriteria]]]:
        wb_list = {}
        removed_wb_list = {}
        exclusion_reasons = {}
        for wb_id, stride_list in preliminary_wb_list.items():
            for rule_name, rule in self.rules or []:
                if not rule.check_include(stride_list):
                    removed_wb_list[wb_id] = stride_list
                    exclusion_reasons[wb_id] = (rule_name, rule)
                    break
            else:
                wb_list[wb_id] = stride_list
        return wb_list, removed_wb_list, exclusion_reasons
