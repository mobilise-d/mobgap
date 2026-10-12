"""Keras-backed wear-time window models."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from importlib.resources import files
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd  # noqa: TC002 - tpcp 2.1 resolves class annotations at runtime.
from tpcp import Algorithm, make_action_safe
from typing_extensions import Self

from mobgap.weartime._model_config import WearTimeModelConfig  # noqa: TC001 - tpcp evaluates annotations.
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
    model_file = files("mobgap.weartime.production_models").joinpath(file_name)
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
    """Inference on lazy raw IMU window batches with standardization inside the fitted Keras model.

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
    model_config: Optional[WearTimeModelConfig]  # noqa: UP045

    def __init__(
        self,
        *,
        model: Any | None = None,
        model_config: WearTimeModelConfig | None = None,
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
