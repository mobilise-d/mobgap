"""Utilities for grouping detected stride candidates in compliant walking bouts."""

from gaitlink.wba._interval_criteria import BaseIntervalCriteria, IntervalDurationCriteria, IntervalParameterCriteria
from gaitlink.wba._stride_selection import StrideSelection
from gaitlink.wba._wb_assembly import WbAssembly
from gaitlink.wba._wb_criteria import LeftRightCriteria, MaxBreakCriteria, NStridesCriteria
from gaitlink.wba._wb_criteria_base import BaseSummaryCriteria, BaseWbCriteria

__all__ = [
    "StrideSelection",
    "BaseIntervalCriteria",
    "IntervalParameterCriteria",
    "IntervalDurationCriteria",
    "LeftRightCriteria",
    "MaxBreakCriteria",
    "NStridesCriteria",
    "BaseWbCriteria",
    "BaseSummaryCriteria",
    "WbAssembly",
]
