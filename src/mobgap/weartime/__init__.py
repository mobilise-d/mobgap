"""Algorithms to detect wear time from raw IMU data."""

from mobgap.weartime._wtd_megaritis_cnn import WtdMegaritis_CNN
from mobgap.weartime._wtd_megaritis_signal import Wtd_Megaritis_signal
from mobgap.weartime._wtd_megaritis_xgboost import WtdMegaritis_XGBoost

__all__ = ["WtdMegaritis_CNN", "WtdMegaritis_XGBoost", "Wtd_Megaritis_signal"]
