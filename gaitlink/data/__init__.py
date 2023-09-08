"""Helper to load data and example in the various supported formats."""

from gaitlink.data._mobilised_matlab_loader import MobilisedMetadata, MobilisedTestData, load_mobilised_matlab_format

__all__ = ["load_mobilised_matlab_format", "MobilisedTestData", "MobilisedMetadata"]
