"""Keras-backed wear-time window models."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003 - tpcp resolves parameter annotations.
from functools import lru_cache
from importlib import import_module
from importlib.resources import files
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd  # noqa: TC002 - tpcp 2.1 resolves class annotations at runtime.
from tpcp import Algorithm, cf, make_action_safe
from typing_extensions import Self

from mobgap.weartime._model_config import WearTimeModelConfig
from mobgap.weartime.utils.ml_feature_extraction import (
    known_window_mask,
    window_count_from_sample_count,
    window_start_end,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from os import PathLike


_MODEL_STANDARDIZATION_LAYER_NAME = "per_window_standardization"


def _as_model_input_sensor_array(sensor_data: np.ndarray) -> np.ndarray:
    return np.asarray(sensor_data, dtype=np.float32, order="C")


@lru_cache(maxsize=1)
def _get_per_window_standardization_layer_class() -> type[Any]:
    """Create and register the serializable Keras per-window standardization layer."""
    tf = import_module("tensorflow")
    layers = import_module("tensorflow.keras.layers")
    register_keras_serializable = import_module("tensorflow.keras.utils").register_keras_serializable

    @register_keras_serializable(package="mobgap")
    class PerWindowStandardization(layers.Layer):  # type: ignore[misc, valid-type]
        def call(self, inputs: Any) -> Any:
            windows = tf.cast(inputs, tf.float32)
            centered = windows - tf.reduce_mean(windows, axis=1, keepdims=True)
            variance = tf.reduce_mean(tf.square(centered), axis=1, keepdims=True)
            variance = tf.maximum(variance, tf.constant(1e-16, dtype=variance.dtype))
            return centered / tf.sqrt(variance)

    return PerWindowStandardization


def _stepped_window_view(sensor_data: np.ndarray, window_samples: int, step_samples: int, n_windows: int) -> np.ndarray:
    if n_windows == 0:
        return np.empty((0, window_samples, sensor_data.shape[1]), dtype=sensor_data.dtype)

    if step_samples == window_samples:
        window_view = sensor_data[: n_windows * window_samples].reshape(
            n_windows,
            window_samples,
            sensor_data.shape[1],
        )
    else:
        window_view = np.lib.stride_tricks.as_strided(
            sensor_data,
            shape=(n_windows, window_samples, sensor_data.shape[1]),
            strides=(step_samples * sensor_data.strides[0], sensor_data.strides[0], sensor_data.strides[1]),
            writeable=False,
        )
    window_view.setflags(write=False)
    return window_view


@lru_cache(maxsize=None)  # noqa: UP033 - Use the Python 3.8-compatible spelling.
def _load_keras_model_resource(file_name: str) -> Any:
    """Load a Keras model from packaged production model resources."""
    model_file = files("mobgap.weartime._megaritis_cnn._models").joinpath(file_name)
    return load_keras_weartime_model(model_file)


def load_keras_weartime_model(model_path: str | PathLike[str]) -> Any:
    """Load a wear-time artifact with a serialized per-window standardization layer."""
    _get_per_window_standardization_layer_class()
    keras_models = import_module("tensorflow.keras.models")
    return keras_models.load_model(model_path)


def _iter_window_batches(
    data: pd.DataFrame,
    config: WearTimeModelConfig,
    *,
    overlap: float,
    window_batch_size: int,
    reference: pd.DataFrame | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    window_samples, step_samples = config.window_parameters(overlap)
    n_windows = window_count_from_sample_count(len(data), window_samples, step_samples)
    window_view = None
    for batch_start in range(0, n_windows, window_batch_size):
        starts = np.arange(batch_start, min(batch_start + window_batch_size, n_windows)) * step_samples
        known = known_window_mask(starts, starts + window_samples, reference)
        if not known.any():
            continue
        # Convert the recording only when the first usable batch is requested.
        if window_view is None:
            sensor_data = (
                data.to_numpy(copy=False)
                if tuple(data.columns) == tuple(config.sensor_cols)
                else data[list(config.sensor_cols)].to_numpy(copy=False)
            )
            sensor_data = _as_model_input_sensor_array(sensor_data)
            window_view = _stepped_window_view(sensor_data, window_samples, step_samples, n_windows)
        batch_window_view = window_view[batch_start : batch_start + len(starts)]
        if not known.all():
            batch_window_view = batch_window_view[known]
        if batch_window_view.flags.c_contiguous:
            windows = batch_window_view
        else:
            windows = np.array(batch_window_view, dtype=np.float32, order="C")
        yield windows, starts[known]


class KerasWeartimeModel(Algorithm):
    """Inference on lazy raw IMU window batches through a supplied Keras model.

    Training and inference pass raw windows to the native graph; it must contain any required preprocessing.
    The built-in Megaritis CNN and CNN-LSTM blueprints include per-window standardization.

    Parameters
    ----------
    model
        Fitted Keras model. Use WtdMegaritisCNNTrainer to train a new model.
    model_config
        Native sampling rate, window duration and ordered input channels produced during training.
    overlap
        Inference window overlap, independent of training overlap.
    prediction_threshold
        Wear probabilities greater than or equal to this threshold become wear predictions.
    predict_batch_size
        Batch size passed to Keras predict.
    window_batch_size
        Maximum number of raw windows prepared at once. No window cache is used.

    Other Parameters
    ----------------
    data
        Recording supplied to run.
    sampling_rate_hz
        Sampling rate of the recording supplied to run.

    Attributes
    ----------
    window_predictions_
        Binary wear predictions per window.
    window_probabilities_
        Wear probabilities per window.
    window_start_end_
        Half-open sample boundaries per window.
    window_samples_
        Window size in samples.
    step_samples_
        Window stride in samples.
    """

    _action_methods = ("run",)
    model: Optional[Any]  # noqa: UP045 - tpcp evaluates annotations at runtime.
    model_config: WearTimeModelConfig

    def __init__(
        self,
        *,
        model: Any | None = None,
        model_config: WearTimeModelConfig = cf(WearTimeModelConfig()),
        overlap: float = 0.75,
        prediction_threshold: float = 0.5,
        predict_batch_size: int = 256,
        window_batch_size: int = 1024,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.overlap = overlap
        self.prediction_threshold = prediction_threshold
        self.predict_batch_size = predict_batch_size
        self.window_batch_size = window_batch_size

    @make_action_safe
    def run(self, data: pd.DataFrame, *, sampling_rate_hz: float) -> Self:
        """Classify one recording at the fitted model's native sampling rate."""
        self.model_config.validate_sampling_rate(sampling_rate_hz)
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.window_samples_, self.step_samples_ = self.model_config.window_parameters(self.overlap)
        self.window_start_end_ = window_start_end(len(data), self.window_samples_, self.step_samples_)
        probability_batches = []
        for windows, _ in _iter_window_batches(
            data, self.model_config, overlap=self.overlap, window_batch_size=self.window_batch_size
        ):
            probabilities = self.model.predict(windows, verbose=0, batch_size=self.predict_batch_size)
            probability_batches.append(np.asarray(probabilities).reshape(-1))
        self.window_probabilities_ = (
            np.concatenate(probability_batches).astype(np.float32, copy=False)
            if probability_batches
            else np.empty(0, dtype=np.float32)
        )
        self.window_predictions_ = (self.window_probabilities_ >= self.prediction_threshold).astype(np.int32)
        return self

    def _create_model(self, config: WearTimeModelConfig) -> Any:  # noqa: ARG002 - Same factory interface as blueprints.
        """Clone a supplied native untrained network; concrete blueprints build their own graph."""
        return import_module("tensorflow").keras.models.clone_model(self.model)


class MegaritisCnnWeartimeModel(KerasWeartimeModel):
    """CNN model blueprint owning its graph structure and input configuration.

    Parameters
    ----------
    model, model_config
        Native fitted network and input configuration. Leave model unset for a fresh training blueprint.
    filters, kernel_size, pool_size
        Filter counts for three convolution blocks, convolution kernel width and max-pooling size.
    dropout_rate, dense_units
        Dropout probability and dense hidden-layer width.
    overlap, prediction_threshold, predict_batch_size, window_batch_size
        Inference execution settings; see KerasWeartimeModel.
    """

    def __init__(
        self,
        *,
        model: Any | None = None,
        model_config: WearTimeModelConfig = cf(WearTimeModelConfig()),
        filters: Sequence[int] = (32, 64, 128),
        kernel_size: int = 9,
        pool_size: int = 2,
        dropout_rate: float = 0.3,
        dense_units: int = 64,
        overlap: float = 0.75,
        prediction_threshold: float = 0.5,
        predict_batch_size: int = 256,
        window_batch_size: int = 1024,
    ) -> None:
        super().__init__(
            model=model,
            model_config=model_config,
            overlap=overlap,
            prediction_threshold=prediction_threshold,
            predict_batch_size=predict_batch_size,
            window_batch_size=window_batch_size,
        )
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units

    def _sequence_layers(self) -> list[Any]:
        layers = import_module("tensorflow.keras.layers")
        return [layers.Flatten()]

    def _create_model(self, config: WearTimeModelConfig) -> Any:
        keras = import_module("tensorflow.keras")
        layers = import_module("tensorflow.keras.layers")
        window_samples, _ = config.window_parameters(0)
        network_layers = [
            layers.Input(shape=(window_samples, len(config.sensor_cols))),
            _get_per_window_standardization_layer_class()(name=_MODEL_STANDARDIZATION_LAYER_NAME),
        ]
        for filters in self.filters:
            network_layers.extend(
                [
                    layers.Conv1D(filters, self.kernel_size, padding="same"),
                    layers.BatchNormalization(),
                    layers.Activation("relu"),
                    layers.MaxPooling1D(self.pool_size),
                    layers.Dropout(self.dropout_rate),
                ]
            )
        network_layers.extend(
            [
                *self._sequence_layers(),
                layers.Dense(self.dense_units),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dropout(self.dropout_rate),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        return keras.Sequential(network_layers)


class MegaritisCnnLstmWeartimeModel(MegaritisCnnWeartimeModel):
    """CNN-LSTM model blueprint with an LSTM between the convolution and dense blocks.

    Parameters
    ----------
    lstm_units
        Width of the LSTM layer.
    model, model_config, filters, kernel_size, pool_size, dropout_rate, dense_units
        Native network, model inputs and shared structural parameters; see MegaritisCnnWeartimeModel.
    overlap, prediction_threshold, predict_batch_size, window_batch_size
        Inference execution settings; see KerasWeartimeModel.
    """

    def __init__(
        self,
        *,
        model: Any | None = None,
        model_config: WearTimeModelConfig = cf(WearTimeModelConfig()),
        filters: Sequence[int] = (32, 64, 128),
        kernel_size: int = 9,
        pool_size: int = 2,
        dropout_rate: float = 0.3,
        dense_units: int = 64,
        lstm_units: int = 64,
        overlap: float = 0.75,
        prediction_threshold: float = 0.5,
        predict_batch_size: int = 256,
        window_batch_size: int = 1024,
    ) -> None:
        super().__init__(
            model=model,
            model_config=model_config,
            filters=filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            dropout_rate=dropout_rate,
            dense_units=dense_units,
            overlap=overlap,
            prediction_threshold=prediction_threshold,
            predict_batch_size=predict_batch_size,
            window_batch_size=window_batch_size,
        )
        self.lstm_units = lstm_units

    def _sequence_layers(self) -> list[Any]:
        layers = import_module("tensorflow.keras.layers")
        return [layers.LSTM(self.lstm_units, return_sequences=False), layers.Dropout(self.dropout_rate)]
