"""Utilities for grouping detected stride candidates in compliant walking bouts."""

from mobgap.wba._interval_criteria import BaseIntervalCriteria, IntervalDurationCriteria, IntervalParameterCriteria
from mobgap.wba._stride_selection import StrideSelection
from mobgap.wba._wb_assembly import WbAssembly
from mobgap.wba._wb_criteria import LeftRightCriteria, MaxBreakCriteria, NStridesCriteria
from mobgap.wba._wb_criteria_base import BaseSummaryCriteria, BaseWbCriteria

__all__ = [
    "BaseIntervalCriteria",
    "BaseSummaryCriteria",
    "BaseWbCriteria",
    "IntervalDurationCriteria",
    "IntervalParameterCriteria",
    "LeftRightCriteria",
    "MaxBreakCriteria",
    "NStridesCriteria",
    "StrideSelection",
    "WbAssembly",
]
