"""Compatibility imports for configurable walking-bout aggregation."""

from mobgap.aggregation._generic_aggregator import GenericAggregator as SDMOAggregator
from mobgap.aggregation._generic_aggregator import _custom_quantile_10, _custom_quantile_90

__all__ = ["SDMOAggregator", "_custom_quantile_10", "_custom_quantile_90"]
