"""Algorithms to detect wear time from raw IMU data."""

from mobgap.weartime._wtd_megaritis_cnn import WtdMegaritisCNN
from mobgap.weartime._wtd_megaritis_signal import WtdMegaritisSignal
from mobgap.weartime._wtd_megaritis_xgboost import WtdMegaritisXGBoost

__all__ = ["WtdMegaritisCNN", "WtdMegaritisSignal", "WtdMegaritisXGBoost"]
