"""Filter and Data Transformation with a familiar algorithm interface."""
from gaitlink.data_transform._cwt import CwtFilter
from gaitlink.data_transform._filter import (
    ButterworthFilter,
    EpflDedriftedGaitFilter,
    EpflDedriftFilter,
    EpflGaitFilter,
    FirFilter,
)
from gaitlink.data_transform._gaussian_filter import GaussianFilter
from gaitlink.data_transform._resample import Resample
from gaitlink.data_transform._savgol_filter import SavgolFilter
from gaitlink.data_transform._utils import chain_transformers

__all__ = [
    "EpflGaitFilter",
    "EpflDedriftFilter",
    "EpflDedriftedGaitFilter",
    "chain_transformers",
    "ButterworthFilter",
    "FirFilter",
    "Resample",
    "CwtFilter",
    "GaussianFilter",
    "SavgolFilter",
]
