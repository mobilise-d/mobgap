"""XGBoost wear-time algorithms, requiring XGBoost at import time."""

from mobgap.weartime._megaritis_xgboost._detector import WtdMegaritisXGBoost
from mobgap.weartime._megaritis_xgboost._trainer import WtdMegaritisXGBoostTrainer

__all__ = ["WtdMegaritisXGBoost", "WtdMegaritisXGBoostTrainer"]
