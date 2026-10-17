"""XGBoost wear-time algorithms, requiring XGBoost at import time."""

from mobgap.weartime.xgboost._detector import WtdMegaritisXGBoost
from mobgap.weartime.xgboost._trainer import WtdMegaritisXGBoostTrainer

__all__ = ["WtdMegaritisXGBoost", "WtdMegaritisXGBoostTrainer"]
