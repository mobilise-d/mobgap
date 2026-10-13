"""Wear-time detection algorithms for identifying when a wearable device is being worn."""

from mobgap.weartime._megaritis_cnn._detector import WtdMegaritisCNN
from mobgap.weartime._megaritis_cnn._model import (
    KerasWeartimeModel,
    MegaritisCnnLstmWeartimeModel,
    MegaritisCnnWeartimeModel,
    load_keras_weartime_model,
)
from mobgap.weartime._megaritis_cnn._trainer import WtdMegaritisCNNTrainer
from mobgap.weartime._megaritis_xgboost._detector import WtdMegaritisXGBoost
from mobgap.weartime._megaritis_xgboost._trainer import WtdMegaritisXGBoostTrainer
from mobgap.weartime._model_config import WearTimeModelConfig
from mobgap.weartime._wtd_megaritis_signal import WtdMegaritisSignal
from mobgap.weartime.base import TrainingData

__all__ = [
    "KerasWeartimeModel",
    "MegaritisCnnLstmWeartimeModel",
    "MegaritisCnnWeartimeModel",
    "TrainingData",
    "WearTimeModelConfig",
    "WtdMegaritisCNN",
    "WtdMegaritisCNNTrainer",
    "WtdMegaritisSignal",
    "WtdMegaritisXGBoost",
    "WtdMegaritisXGBoostTrainer",
    "load_keras_weartime_model",
]
