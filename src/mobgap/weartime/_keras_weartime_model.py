"""Keras-backed wear-time window models."""

from __future__ import annotations

import logging
from contextlib import suppress
from functools import lru_cache
from importlib import import_module
from importlib.resources import files
from math import ceil
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

import numpy as np
import pandas as pd  # noqa: TC002 - tpcp 2.1 resolves class annotations at runtime.
from tpcp import Algorithm, OptimizableParameter, make_action_safe, make_optimize_safe
from tpcp.misc import classproperty
from typing_extensions import Self

from mobgap.consts import BF_SENSOR_COLS

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from os import PathLike

    from mobgap.weartime.base import RecordingSampleCounts, TrainingData

_LOGGER = logging.getLogger(__name__)
_C = TypeVar("_C", bound=Callable[..., Any])
_MODEL_STANDARDIZATION_LAYER_NAME = "per_window_standardization"


def _make_action_safe(action_method: _C) -> _C:
    """Apply tpcp action checks while staying compatible with tpcp 2.1's test mixin."""
    safe_action_method = make_action_safe(action_method)
    with suppress(AttributeError):
        delattr(safe_action_method, "__tpcp_action_method")
    return safe_action_method


def _rss_mb() -> float | None:
    try:
        psutil = import_module("psutil")
    except ImportError:
        return None
    return float(psutil.Process().memory_info().rss / 1024**2)


def _validate_model_sampling_rate(
    *,
    trained_sampling_rate_hz: float | None,
    sampling_rate_hz: float,
    allow_mismatch: bool,
) -> None:
    if trained_sampling_rate_hz is None or allow_mismatch:
        return
    if not np.isclose(float(trained_sampling_rate_hz), float(sampling_rate_hz)):
        raise ValueError(
            "The Keras wear-time model was trained for "
            f"{trained_sampling_rate_hz} Hz, but inference received {sampling_rate_hz} Hz. "
            "Pass a model trained at the target sampling rate or set `allow_sampling_rate_mismatch=True`."
        )


def _window_start_end(n_samples: int, window_samples: int, step_samples: int) -> np.ndarray:
    if n_samples < window_samples:
        return np.empty((0, 2), dtype=np.int64)
    starts = np.arange(0, n_samples - window_samples + 1, step_samples, dtype=np.int64)
    return np.column_stack([starts, starts + window_samples])


def _window_count_from_sample_count(n_samples: int, window_samples: int, step_samples: int) -> int:
    if n_samples < 0:
        raise ValueError("Recording sample counts must be non-negative.")
    if n_samples < window_samples:
        return 0
    return (n_samples - window_samples) // step_samples + 1


def _steps_per_epoch_from_recording_sample_counts(
    recording_sample_counts: RecordingSampleCounts,
    *,
    window_samples: int,
    step_samples: int,
    batch_size: int,
) -> tuple[int, int]:
    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")

    total_windows = sum(
        _window_count_from_sample_count(int(sample_count), window_samples, step_samples)
        for sample_count in recording_sample_counts
    )
    if total_windows <= 0:
        raise ValueError("The training recordings do not contain any full model windows.")
    return total_windows, ceil(total_windows / batch_size)


def _standardize_windows(windows: np.ndarray) -> np.ndarray:
    # The caller passes freshly materialized window batches, so standardize in place to avoid extra copies.
    windows = windows.astype(np.float32, copy=False)
    mean = windows.mean(axis=1, keepdims=True)
    windows -= mean
    variance = np.einsum("ijk,ijk->ik", windows, windows, optimize=True)[:, None, :] / windows.shape[1]
    variance[variance < 1e-16] = 1e-16
    windows /= np.sqrt(variance, dtype=np.float32)
    return windows


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


def _reference_weartime_interval_arrays(reference_weartime: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if len(reference_weartime) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    intervals = reference_weartime[["start", "end"]].to_numpy(dtype=np.int64, copy=False)
    intervals = intervals[np.argsort(intervals[:, 0], kind="stable")]
    return intervals[:, 0], intervals[:, 1]


def _labels_from_interval_arrays(
    centers: np.ndarray, interval_starts: np.ndarray, interval_ends: np.ndarray
) -> np.ndarray:
    if len(interval_starts) == 0:
        return np.zeros(len(centers), dtype=np.int32)

    interval_indices = np.searchsorted(interval_starts, centers, side="right") - 1
    labels = np.zeros(len(centers), dtype=np.int32)
    valid = interval_indices >= 0
    labels[valid] = centers[valid] < interval_ends[interval_indices[valid]]
    return labels


def _labels_from_interval_centers(centers: np.ndarray, reference_weartime: pd.DataFrame) -> np.ndarray:
    return _labels_from_interval_arrays(centers, *_reference_weartime_interval_arrays(reference_weartime))


@lru_cache(maxsize=None)  # noqa: UP033 - Use the Python 3.8-compatible spelling.
def _load_keras_model_resource(file_name: str) -> Any:
    """Load a Keras model from packaged production model resources."""
    model_file = files("mobgap.weartime.production_models").joinpath(file_name)
    return load_keras_weartime_model(model_file)


def load_keras_weartime_model(model_path: str | PathLike[str]) -> Any:
    """Load a wear-time Keras artifact, registering the optional model standardization layer first."""
    _get_per_window_standardization_layer_class()
    keras_models = import_module("tensorflow.keras.models")
    return keras_models.load_model(model_path)


class BaseKerasWeartimeModel(Algorithm):
    """Base class for Keras models that classify wear-time windows."""

    _action_methods = ("run",)

    _model: OptimizableParameter[Optional[Any]]  # noqa: UP045 - tpcp 2.1 needs Python 3.9-evaluable strings.
    _trained_sampling_rate_hz: OptimizableParameter[Optional[float]]  # noqa: UP045

    data: pd.DataFrame
    sampling_rate_hz: float
    window_predictions_: np.ndarray
    window_probabilities_: np.ndarray
    window_start_end_: np.ndarray
    window_samples_: int
    step_samples_: int

    def __init__(
        self,
        *,
        _model: Any | None = None,
        _trained_sampling_rate_hz: float | None = None,
        window_sec: float = 5.0,
        overlap: float = 0.75,
        sensor_cols: Sequence[str] = tuple(BF_SENSOR_COLS),
        prediction_threshold: float = 0.5,
        batch_size: int = 1024,
        epochs: int = 1,
        fit_verbose: int = 1,
        predict_batch_size: int = 256,
        window_batch_size: int = 1024,
        shuffle_buffer_size: int = 8192,
        allow_sampling_rate_mismatch: bool = False,
        standardize_in_model: bool = False,
    ) -> None:
        self._model = _model
        self._trained_sampling_rate_hz = _trained_sampling_rate_hz
        self.window_sec = window_sec
        self.overlap = overlap
        self.sensor_cols = sensor_cols
        self.prediction_threshold = prediction_threshold
        self.batch_size = batch_size
        self.epochs = epochs
        self.fit_verbose = fit_verbose
        self.predict_batch_size = predict_batch_size
        self.window_batch_size = window_batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.allow_sampling_rate_mismatch = allow_sampling_rate_mismatch
        self.standardize_in_model = standardize_in_model

    @_make_action_safe
    def run(self, data: pd.DataFrame, *, sampling_rate_hz: float) -> Self:
        """Classify all model windows in a single recording."""
        if self._model is None:
            raise RuntimeError("Keras model is not trained or loaded. Call `self_optimize` or load a pretrained model.")

        _validate_model_sampling_rate(
            trained_sampling_rate_hz=self._trained_sampling_rate_hz,
            sampling_rate_hz=sampling_rate_hz,
            allow_mismatch=self.allow_sampling_rate_mismatch,
        )

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.window_samples_, self.step_samples_ = self._window_parameters(sampling_rate_hz)
        self.window_start_end_ = _window_start_end(len(data), self.window_samples_, self.step_samples_)

        probability_batches: list[np.ndarray] = []
        for batch_index, (windows, _) in enumerate(self._iter_window_batches(data, sampling_rate_hz)):
            _LOGGER.debug(
                "Predicting Keras wear-time window batch %s: n_windows=%s, rss_mb=%s",
                batch_index,
                len(windows),
                _rss_mb(),
            )
            probabilities = self._model.predict(windows, verbose=0, batch_size=self.predict_batch_size)
            probability_batches.append(np.asarray(probabilities).reshape(-1))

        self.window_probabilities_ = (
            np.concatenate(probability_batches).astype(np.float32, copy=False)
            if probability_batches
            else np.empty(0, dtype=np.float32)
        )
        self.window_predictions_ = (self.window_probabilities_ > self.prediction_threshold).astype(np.int32)
        return self

    @make_optimize_safe
    def self_optimize(
        self,
        training_data: TrainingData,
        *,
        sampling_rate_hz: float,
        recording_sample_counts: RecordingSampleCounts,
    ) -> Self:
        """Train the internal Keras model from lazy recording-level training data."""
        if self._model is not None:
            raise RuntimeError("Keras model is already trained or loaded. Initialize an untrained model first.")

        epochs = int(self.epochs)
        if epochs > 1 and iter(training_data) is training_data:
            raise ValueError(
                "`training_data` must be re-iterable for multi-epoch training. "
                "Pass a lazy dataset wrapper whose `__iter__` creates a fresh recording iterator."
            )

        window_samples, step_samples = self._window_parameters(sampling_rate_hz)
        batch_size = int(self.batch_size)
        recording_sample_counts = tuple(int(sample_count) for sample_count in recording_sample_counts)
        total_windows, steps_per_epoch = _steps_per_epoch_from_recording_sample_counts(
            recording_sample_counts,
            window_samples=window_samples,
            step_samples=step_samples,
            batch_size=batch_size,
        )
        self._trained_sampling_rate_hz = float(sampling_rate_hz)
        self._model = self._create_model((window_samples, len(self.sensor_cols)))
        fit_parameters: dict[str, Any] = {
            "epochs": epochs,
            "verbose": self.fit_verbose,
            "steps_per_epoch": steps_per_epoch,
        }

        _LOGGER.debug(
            "Starting Keras wear-time model training: sampling_rate_hz=%s, epochs=%s, batch_size=%s, "
            "window_batch_size=%s, shuffle_buffer_size=%s, total_windows=%s, steps_per_epoch=%s, rss_mb=%s",
            sampling_rate_hz,
            fit_parameters["epochs"],
            batch_size,
            self.window_batch_size,
            self.shuffle_buffer_size,
            total_windows,
            steps_per_epoch,
            _rss_mb(),
        )

        self._model.fit(
            self._make_tf_dataset(
                training_data,
                sampling_rate_hz=sampling_rate_hz,
                batch_size=batch_size,
                window_samples=window_samples,
            ),
            **fit_parameters,
        )
        return self

    def _create_model(self, input_shape: tuple[int, int]) -> Any:
        raise NotImplementedError

    def _model_input_layers(self, input_shape: tuple[int, int]) -> list[Any]:
        layers = import_module("tensorflow.keras.layers")
        input_layers = [layers.Input(shape=input_shape)]
        if self.standardize_in_model:
            input_layers.append(_get_per_window_standardization_layer_class()(name=_MODEL_STANDARDIZATION_LAYER_NAME))
        return input_layers

    def _window_parameters(self, sampling_rate_hz: float) -> tuple[int, int]:
        if not 0 <= self.overlap < 1:
            raise ValueError("`overlap` must be in [0, 1).")
        window_samples = int(self.window_sec * sampling_rate_hz)
        overlap_samples = int(window_samples * self.overlap)
        step_samples = window_samples - overlap_samples
        if window_samples <= 0:
            raise ValueError("`window_sec` and `sampling_rate_hz` must result in at least one sample per window.")
        if step_samples <= 0:
            raise ValueError("The window step must be positive. Check `window_sec` and `overlap`.")
        return window_samples, step_samples

    def _iter_window_batches(
        self,
        data: pd.DataFrame,
        sampling_rate_hz: float,
        reference_weartime: pd.DataFrame | None = None,
        recording_index: int | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray | None]]:
        window_samples, step_samples = self._window_parameters(sampling_rate_hz)
        window_start_end = _window_start_end(len(data), window_samples, step_samples)
        _LOGGER.debug(
            "Preparing Keras wear-time windows for recording %s: n_samples=%s, n_windows=%s, rss_mb=%s",
            recording_index,
            len(data),
            len(window_start_end),
            _rss_mb(),
        )
        if len(window_start_end) == 0:
            return

        sensor_data = (
            data.to_numpy(copy=False)
            if tuple(data.columns) == tuple(self.sensor_cols)
            else data[list(self.sensor_cols)].to_numpy(copy=False)
        )
        sensor_data = _as_model_input_sensor_array(sensor_data)
        window_view = _stepped_window_view(sensor_data, window_samples, step_samples, len(window_start_end))
        reference_interval_arrays = (
            _reference_weartime_interval_arrays(reference_weartime) if reference_weartime is not None else None
        )

        for batch_index, batch_start in enumerate(range(0, len(window_start_end), self.window_batch_size)):
            batch_start_end = window_start_end[batch_start : batch_start + self.window_batch_size]
            starts = batch_start_end[:, 0]
            batch_window_view = window_view[batch_start : batch_start + len(batch_start_end)]
            if self.standardize_in_model and batch_window_view.flags.c_contiguous:
                windows = batch_window_view
            else:
                windows = np.array(batch_window_view, dtype=np.float32, order="C")
            if not self.standardize_in_model:
                windows = _standardize_windows(windows)
            labels = (
                _labels_from_interval_arrays(starts + window_samples // 2, *reference_interval_arrays)
                if reference_interval_arrays is not None
                else None
            )
            _LOGGER.debug(
                "Yielding Keras wear-time window batch %s for recording %s: start_window=%s, n_windows=%s, rss_mb=%s",
                batch_index,
                recording_index,
                batch_start,
                len(windows),
                _rss_mb(),
            )
            yield windows, labels

    def _iter_training_window_batches(
        self, training_data: TrainingData, sampling_rate_hz: float
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for recording_index, (data, reference_weartime) in enumerate(training_data):
            _LOGGER.debug(
                "Loaded Keras wear-time training recording %s: n_samples=%s, n_reference_intervals=%s, rss_mb=%s",
                recording_index,
                len(data),
                len(reference_weartime),
                _rss_mb(),
            )
            for windows, labels in self._iter_window_batches(
                data,
                sampling_rate_hz,
                reference_weartime=reference_weartime,
                recording_index=recording_index,
            ):
                if labels is not None:
                    yield windows, labels

    def _make_tf_dataset(
        self,
        training_data: TrainingData,
        *,
        sampling_rate_hz: float,
        batch_size: int,
        window_samples: int,
    ) -> Any:
        tf = import_module("tensorflow")
        output_signature = (
            tf.TensorSpec(shape=(None, window_samples, len(self.sensor_cols)), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )
        dataset = tf.data.Dataset.from_generator(
            lambda: self._iter_training_window_batches(training_data, sampling_rate_hz),
            output_signature=output_signature,
        )
        dataset = dataset.unbatch()
        if self.shuffle_buffer_size > 0:
            dataset = dataset.shuffle(self.shuffle_buffer_size, reshuffle_each_iteration=True)
        # Keras keeps consuming the same dataset iterator across epoch boundaries when ``steps_per_epoch`` is set.
        # Repeat after batching so every epoch is one full pass through the same batch sequence, including a final
        # partial batch if the number of windows is not divisible by ``batch_size``.
        return dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)


class MegaritisCnnWeartimeModel(BaseKerasWeartimeModel):
    """1D CNN wear-time model with per-window standardization.

    Parameters
    ----------
    _model
        The underlying Keras model. If ``None``, ``self_optimize`` creates and trains a new model.
    _trained_sampling_rate_hz
        Sampling rate used when the model was loaded or trained. Inference rejects mismatching sampling rates by
        default.
    window_sec
        Window size in seconds.
    overlap
        Fractional window overlap.
    sensor_cols
        Body-frame sensor columns passed to the Keras model.
    prediction_threshold
        Probability threshold used to convert Keras probabilities to binary window labels.
    batch_size
        Keras training batch size.
    epochs
        Default number of training epochs.
    fit_verbose
        Verbosity passed to Keras ``fit`` during self-optimization.
    predict_batch_size
        Keras prediction batch size.
    window_batch_size
        Number of windows materialized and standardized at once while iterating over a recording.
    shuffle_buffer_size
        Buffer size used to shuffle training windows in the TensorFlow dataset.
    allow_sampling_rate_mismatch
        If ``True``, allow inference at a different sampling rate than the loaded or trained model.
    standardize_in_model
        If ``True``, add a Keras preprocessing layer that standardizes each window per channel and skip NumPy-side
        standardization while iterating windows. Keep ``False`` for existing pretrained models that already expect
        standardized inputs.
    filters
        Number of convolution filters in the three CNN blocks.
    kernel_size
        Kernel size of the convolution layers.
    pool_size
        Pool size of the max-pooling layers.
    dropout_rate
        Dropout rate used after each model block.
    dense_units
        Number of units in the dense hidden layer.
    learning_rate
        Adam optimizer learning rate.

    Other Parameters
    ----------------
    data
        The raw IMU data passed to ``run``.
    sampling_rate_hz
        The sampling rate of ``data`` passed to ``run``.

    Attributes
    ----------
    window_predictions_
        Binary window labels predicted by the Keras model.
    window_probabilities_
        Wear probabilities predicted by the Keras model.
    window_start_end_
        Array of ``[start, end)`` sample boundaries for each classified window.
    window_samples_
        Window size in samples used for the last call.
    step_samples_
        Step size in samples used for the last call.
    """

    class PredefinedParameters:
        """Predefined parameters for the production CNN wear-time model."""

        @classproperty
        def lowback(cls) -> MappingProxyType:  # noqa: N805
            return MappingProxyType(
                {
                    "_model": _load_keras_model_resource("cnn_lowback_model.keras"),
                    "_trained_sampling_rate_hz": 100.0,
                }
            )

    def __init__(
        self,
        *,
        _model: Any | None = None,
        _trained_sampling_rate_hz: float | None = None,
        window_sec: float = 5.0,
        overlap: float = 0.75,
        sensor_cols: Sequence[str] = tuple(BF_SENSOR_COLS),
        prediction_threshold: float = 0.5,
        batch_size: int = 1024,
        epochs: int = 60,
        fit_verbose: int = 1,
        predict_batch_size: int = 256,
        window_batch_size: int = 1024,
        shuffle_buffer_size: int = 8192,
        allow_sampling_rate_mismatch: bool = False,
        standardize_in_model: bool = False,
        filters: Sequence[int] = (32, 64, 128),
        kernel_size: int = 9,
        pool_size: int = 2,
        dropout_rate: float = 0.3,
        dense_units: int = 64,
        learning_rate: float = 0.001,
    ) -> None:
        super().__init__(
            _model=_model,
            _trained_sampling_rate_hz=_trained_sampling_rate_hz,
            window_sec=window_sec,
            overlap=overlap,
            sensor_cols=sensor_cols,
            prediction_threshold=prediction_threshold,
            batch_size=batch_size,
            epochs=epochs,
            fit_verbose=fit_verbose,
            predict_batch_size=predict_batch_size,
            window_batch_size=window_batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            allow_sampling_rate_mismatch=allow_sampling_rate_mismatch,
            standardize_in_model=standardize_in_model,
        )
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.learning_rate = learning_rate

    def _create_model(self, input_shape: tuple[int, int]) -> Any:
        keras = import_module("tensorflow.keras")
        layers = import_module("tensorflow.keras.layers")

        model = keras.Sequential(
            [
                *self._model_input_layers(input_shape),
                layers.Conv1D(self.filters[0], self.kernel_size, padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling1D(self.pool_size),
                layers.Dropout(self.dropout_rate),
                layers.Conv1D(self.filters[1], self.kernel_size, padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling1D(self.pool_size),
                layers.Dropout(self.dropout_rate),
                layers.Conv1D(self.filters[2], self.kernel_size, padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling1D(self.pool_size),
                layers.Dropout(self.dropout_rate),
                layers.Flatten(),
                layers.Dense(self.dense_units),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dropout(self.dropout_rate),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model


class MegaritisCnnLstmWeartimeModel(MegaritisCnnWeartimeModel):
    """CNN-LSTM wear-time model with per-window standardization.

    Parameters
    ----------
    _model
        The underlying Keras model. If ``None``, ``self_optimize`` creates and trains a new model.
    _trained_sampling_rate_hz
        Sampling rate used when the model was loaded or trained. Inference rejects mismatching sampling rates by
        default.
    window_sec
        Window size in seconds.
    overlap
        Fractional window overlap.
    sensor_cols
        Body-frame sensor columns passed to the Keras model.
    prediction_threshold
        Probability threshold used to convert Keras probabilities to binary window labels.
    batch_size
        Keras training batch size.
    epochs
        Default number of training epochs.
    fit_verbose
        Verbosity passed to Keras ``fit`` during self-optimization.
    predict_batch_size
        Keras prediction batch size.
    window_batch_size
        Number of windows materialized and standardized at once while iterating over a recording.
    shuffle_buffer_size
        Buffer size used to shuffle training windows in the TensorFlow dataset.
    allow_sampling_rate_mismatch
        If ``True``, allow inference at a different sampling rate than the loaded or trained model.
    standardize_in_model
        If ``True``, add a Keras preprocessing layer that standardizes each window per channel and skip NumPy-side
        standardization while iterating windows. Keep ``False`` for existing pretrained models that already expect
        standardized inputs.
    filters
        Number of convolution filters in the three CNN blocks.
    kernel_size
        Kernel size of the convolution layers.
    pool_size
        Pool size of the max-pooling layers.
    dropout_rate
        Dropout rate used after each model block.
    dense_units
        Number of units in the dense hidden layer.
    learning_rate
        Adam optimizer learning rate.
    lstm_units
        Number of units in the LSTM layer.

    Other Parameters
    ----------------
    data
        The raw IMU data passed to ``run``.
    sampling_rate_hz
        The sampling rate of ``data`` passed to ``run``.

    Attributes
    ----------
    window_predictions_
        Binary window labels predicted by the Keras model.
    window_probabilities_
        Wear probabilities predicted by the Keras model.
    window_start_end_
        Array of ``[start, end)`` sample boundaries for each classified window.
    window_samples_
        Window size in samples used for the last call.
    step_samples_
        Step size in samples used for the last call.
    """

    class PredefinedParameters:
        """Predefined parameters for the production CNN-LSTM wear-time model."""

        @classproperty
        def lowback(cls) -> MappingProxyType:  # noqa: N805
            return MappingProxyType(
                {
                    "_model": _load_keras_model_resource("cnn_lstm_lowback_model.keras"),
                    "_trained_sampling_rate_hz": 100.0,
                }
            )

    def __init__(
        self,
        *,
        _model: Any | None = None,
        _trained_sampling_rate_hz: float | None = None,
        window_sec: float = 5.0,
        overlap: float = 0.75,
        sensor_cols: Sequence[str] = tuple(BF_SENSOR_COLS),
        prediction_threshold: float = 0.5,
        batch_size: int = 1024,
        epochs: int = 57,
        fit_verbose: int = 1,
        predict_batch_size: int = 256,
        window_batch_size: int = 1024,
        shuffle_buffer_size: int = 8192,
        allow_sampling_rate_mismatch: bool = False,
        standardize_in_model: bool = False,
        filters: Sequence[int] = (32, 64, 128),
        kernel_size: int = 9,
        pool_size: int = 2,
        dropout_rate: float = 0.3,
        dense_units: int = 64,
        learning_rate: float = 0.001,
        lstm_units: int = 64,
    ) -> None:
        super().__init__(
            _model=_model,
            _trained_sampling_rate_hz=_trained_sampling_rate_hz,
            window_sec=window_sec,
            overlap=overlap,
            sensor_cols=sensor_cols,
            prediction_threshold=prediction_threshold,
            batch_size=batch_size,
            epochs=epochs,
            fit_verbose=fit_verbose,
            predict_batch_size=predict_batch_size,
            window_batch_size=window_batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            allow_sampling_rate_mismatch=allow_sampling_rate_mismatch,
            standardize_in_model=standardize_in_model,
            filters=filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            dropout_rate=dropout_rate,
            dense_units=dense_units,
            learning_rate=learning_rate,
        )
        self.lstm_units = lstm_units

    def _create_model(self, input_shape: tuple[int, int]) -> Any:
        keras = import_module("tensorflow.keras")
        layers = import_module("tensorflow.keras.layers")

        model = keras.Sequential(
            [
                *self._model_input_layers(input_shape),
                layers.Conv1D(self.filters[0], self.kernel_size, padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling1D(self.pool_size),
                layers.Dropout(self.dropout_rate),
                layers.Conv1D(self.filters[1], self.kernel_size, padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling1D(self.pool_size),
                layers.Dropout(self.dropout_rate),
                layers.Conv1D(self.filters[2], self.kernel_size, padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling1D(self.pool_size),
                layers.Dropout(self.dropout_rate),
                layers.LSTM(self.lstm_units, return_sequences=False),
                layers.Dropout(self.dropout_rate),
                layers.Dense(self.dense_units),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dropout(self.dropout_rate),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model
