"""Wear-time detection algorithms for identifying when a wearable device is being worn."""

from mobgap.weartime._keras_weartime_model import (
    KerasWeartimeModel,
    MegaritisCnnLstmWeartimeModel,
    MegaritisCnnWeartimeModel,
    load_keras_weartime_model,
)
from mobgap.weartime._model_config import WearTimeModelConfig
from mobgap.weartime._wtd_megaritis_cnn import WtdMegaritisCNN
from mobgap.weartime._wtd_megaritis_cnn_trainer import WtdMegaritisCNNTrainer
from mobgap.weartime._wtd_megaritis_signal import WtdMegaritisSignal
from mobgap.weartime._wtd_megaritis_xgboost import WtdMegaritisXGBoost
from mobgap.weartime._wtd_megaritis_xgboost_trainer import WtdMegaritisXGBoostTrainer
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
