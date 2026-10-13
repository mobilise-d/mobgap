"""Wear-time signal detection and shared model input and training interfaces."""

from mobgap.weartime._model_config import WearTimeModelConfig
from mobgap.weartime._wtd_megaritis_signal import WtdMegaritisSignal
from mobgap.weartime.base import TrainingData

__all__ = ["TrainingData", "WearTimeModelConfig", "WtdMegaritisSignal"]
