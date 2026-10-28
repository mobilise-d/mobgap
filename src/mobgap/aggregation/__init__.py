"""Algorithms to aggregate results across walking bouts."""

from mobgap.aggregation._generic_aggregator import GenericAggregator
from mobgap.aggregation._mobilised_aggregator import MobilisedAggregator
from mobgap.aggregation._threshold_check import apply_thresholds, get_mobilised_dmo_thresholds

SDMOAggregator = GenericAggregator

__all__ = [
    "GenericAggregator",
    "MobilisedAggregator",
    "SDMOAggregator",
    "apply_thresholds",
    "get_mobilised_dmo_thresholds",
]
