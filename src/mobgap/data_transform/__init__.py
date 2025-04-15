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
    "ButterworthFilter",
    "Crop",
    "CwtFilter",
    "EpflDedriftFilter",
    "EpflDedriftedGaitFilter",
    "EpflGaitFilter",
    "FirFilter",
    "GaussianFilter",
    "HampelFilter",
    "Pad",
    "Resample",
    "SavgolFilter",
    "chain_transformers",
]
