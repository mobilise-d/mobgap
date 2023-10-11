"""Utilities for grouping detected stride candidates in compliant walking bouts."""
from gaitlink.wba._interval_criteria import BaseIntervalCriteria, IntervalDurationCriteria, IntervalParameterCriteria
from gaitlink.wba._stride_selection import StrideSelection, default_mobilised_stride_selection_rules

__all__ = [
    "StrideSelection",
    "BaseIntervalCriteria",
    "IntervalParameterCriteria",
    "IntervalDurationCriteria",
    "default_mobilised_stride_selection_rules",
]
