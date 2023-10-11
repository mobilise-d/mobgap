import uuid
from typing import Optional

import numpy as np
import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self

from gaitlink.wba._wb_criteria_base import BaseWBCriteria, EndOfList


class WBAssembly(Algorithm):
    """Assembles strides into walking bouts based on a set of criteria.

    This method uses a two-step approach.
    First, it iterates through the list of strides and adds them to a preliminary walking bout until a termination
    is reached.
    These preliminary walking bouts are then checked against a set of inclusion rules.
    See Notes sections for some more details.

    Parameters
    ----------
    rules
        The rules that are used to assemble the walking bouts.
        They need to be provided as a list of tuples, where the first element is the name of the rule and the second
        is an instance of a subclass of :class:`gaitlink.wba.BaseWBCriteria`.
        If `None`, no rules are applied and all strides are returned as a single WB.

    Attributes
    ----------
    annotated_stride_list_
        A dataframe containing all strides that are considered that are part of a WB.
        The dataframe has an additional column `wb_id` that contains the id of the WB the stride is part of.
        This output can be used to calculate per-WB statistics, by grouping by the `wb_id` column.
    wbs_
        A dictionary containing all walking bouts that were assembled.
        The keys are the ids of the walking bouts.
        The values are dataframes containing the strides that are part of the respective walking bout.
    excluded_stride_list_
        A dataframe containing all strides that are considered invalid, as they are not part of a WB.
        This can happen, because strides were part of a preliminary WB that was later discarded or because they were
        never part of a WB, which can happen when a termination rule moves the start or end of a WB.
    excluded_wbs_
        A dictionary containing all walking bouts that were discarded.
    termination_reasons_
        A dictionary containing the reason why a WB was terminated.
        The dictionary has the WB id as key and a tuple of the rule name and the rule object as value.
        These rules are match to the rule tuples provided in the `rules` parameter.
        The only exception is the `end_of_list` rule, which is used when the end of the stride list is reached and
        no other rule terminated the WB.
        This dictionary contains all WBs, including the ones that were discarded.
    exclusion_reasons_
        A dictionary containing the reason why a stride was excluded.
        The dictionary has the stride id as key and a tuple of the rule name and the rule object as value.
        This dictionary only contains WBs that were discarded.

    Other Parameters
    ----------------
    stride_list
        The stride list provided to the :meth:`assemble` method.

    Notes
    -----
    Each rule can act as a termination rule and an inclusion rule using the `check_wb_start_end` and `check_include`
    methods respectively.
    Inclusion rules are rather simple, as they only need to return a boolean value and can perform a simple check on
    the stride list or a preliminary WB.
    Termination rules are more complex, as they need to be able to not just terminate a WB, but also adjust the start
    and end of the WB.
    For example, the break rule does not just stop a WB, when there is a gap between strides, it also removes the
    last stride of the WB, as this is not considered a valid stride based on the Mobilise-D definition.
    This means, it must also signal that the last stride should be removed.
    This is done by providing all strides to the `check_wb_start_end` method and the current start and end of the WB as
    defined by all previous rules.
    The method can then adjust the start and end of the WB to implement arbitrary complex rules.
    At each step, the rule that would result in the shortest WB is chosen (i.e. the latest start and the earliest end).

    For the currently implemented rules, a lot of this complexity is not needed.
    However, the current approach was developed as a general framework to solve these types of grouping issues.
    More complex rules where implemented, but ultimately not used, as they were not needed for the Mobilise-D pipeline.
    However, we still left the basic framework in place to allow for more complex rules to be implemented in the future
    or on the user side.

    """

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
        for _, rule in self.rules or []:
            if not isinstance(rule, BaseWBCriteria):
                raise TypeError("All rules must be instances of `WBCriteria` or one of its child classes.")

        self.stride_list = stride_list
        stride_list_sorted = self.stride_list.sort_values(by=["start", "end"])

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
