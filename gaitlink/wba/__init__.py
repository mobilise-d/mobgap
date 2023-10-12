"""Utilities for grouping detected stride candidates in compliant walking bouts."""
from gaitlink.wba._interval_criteria import BaseIntervalCriteria, IntervalDurationCriteria, IntervalParameterCriteria
from gaitlink.wba._stride_selection import StrideSelection, default_mobilised_stride_selection_rules
from gaitlink.wba._wb_assembly import WBAssembly, default_mobilised_wb_rules
from gaitlink.wba._wb_criteria import LeftRightCriteria, MaxBreakCriteria, NStridesCriteria
from gaitlink.wba._wb_criteria_base import BaseSummaryCriteria, BaseWBCriteria

__all__ = [
    "StrideSelection",
    "BaseIntervalCriteria",
    "IntervalParameterCriteria",
    "IntervalDurationCriteria",
    "default_mobilised_stride_selection_rules",
    "LeftRightCriteria",
    "MaxBreakCriteria",
    "NStridesCriteria",
    "BaseWBCriteria",
    "BaseSummaryCriteria",
    "WBAssembly",
    "default_mobilised_wb_rules",
]
