"""Filter and Data Transformation with a familiar algorithm interface."""

from mobgap.data_transform._cwt import CwtFilter
from mobgap.data_transform._filter import (
    ButterworthFilter,
    EpflDedriftedGaitFilter,
    EpflDedriftFilter,
    EpflGaitFilter,
    FirFilter,
    HampelFilter,
)
from mobgap.data_transform._gaussian_filter import GaussianFilter
from mobgap.data_transform._padding import Crop, Pad
from mobgap.data_transform._resample import Resample
from mobgap.data_transform._savgol_filter import SavgolFilter
from mobgap.data_transform._utils import chain_transformers

__all__ = [
    "EpflGaitFilter",
    "EpflDedriftFilter",
    "EpflDedriftedGaitFilter",
    "chain_transformers",
    "ButterworthFilter",
    "FirFilter",
    "HampelFilter",
    "Resample",
    "CwtFilter",
    "GaussianFilter",
    "SavgolFilter",
    "Pad",
    "Crop",
]
