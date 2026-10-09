# Copyright 2026 Dr Dimitrios Megaritis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import pickle
from collections.abc import Iterator, Sequence  # noqa: TC003 - tpcp resolves algorithm annotations at runtime.
from datetime import time
from functools import lru_cache
from importlib import import_module
from importlib.resources import files
from types import MappingProxyType
from typing import Any, Final, Literal, Optional, Protocol

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from sklearn.utils.validation import check_is_fitted
from tpcp import OptimizableParameter, make_action_safe, make_optimize_safe
from tpcp.caching import hybrid_cache
from tpcp.misc import classproperty
from typing_extensions import Self, TypedDict, Unpack

from mobgap._utils_internal.misc import timed_action_method
from mobgap.consts import BF_SENSOR_COLS
from mobgap.utils.array_handling import bool_array_to_start_end_array
from mobgap.utils.dtypes import assert_is_sensor_data
from mobgap.weartime._optimization_presets import _sustain_weartime_optimization_defaults
from mobgap.weartime.base import (
    BaseWeartimeDetector,
    TrainingData,
    _unify_weartime_df,
    base_weartime_docfiller,
)
from mobgap.weartime.utils._intervals import _validate_waking_hours, _with_local_datetimes
from mobgap.weartime.utils.feature_extraction import (
    FEATURE_ORDER_90PCT,
    FULL_FEATURE_ORDER,
    extract_features_batched,
)
from mobgap.weartime.utils.ml_feature_extraction import (
    labels_from_interval_centers,
    window_start_end,
)
from mobgap.weartime.utils.windows_to_weartime import (
    filter_short_wear_bouts_by_confidence,
    overlapping_window_predictions_to_sample_labels,
    remove_isolated_short_periods_from_intervals,
)


class _SklearnWearTimeClassifier(Protocol):
    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs: Any) -> Self: ...  # noqa: N803

    def predict_proba(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray: ...  # noqa: N803


@lru_cache(maxsize=None)  # noqa: UP033 - Use the Python 3.8-compatible spelling.
def _load_pickle_resource(file_name: str) -> Any:
    resource = files("mobgap.weartime.production_models").joinpath(file_name)
    try:
        with resource.open("rb") as file:
            return pickle.load(file)
    except ModuleNotFoundError as exc:
        if exc.name == "xgboost":
            raise ImportError(
                "The optional dependency `xgboost` is required to load the pretrained XGBoost wear-time models. "
                "Install MobGap with the `weartime` extra to use `WtdMegaritisXGBoost`."
            ) from exc
        raise


def _new_xgboost_classifier() -> _SklearnWearTimeClassifier:
    try:
        xgboost = import_module("xgboost")
    except ImportError as exc:
        raise ImportError(
            "The optional dependency `xgboost` is required to train an XGBoost wear-time model. "
            "Install MobGap with the `weartime` extra to use `WtdMegaritisXGBoost.self_optimize`."
        ) from exc
    return xgboost.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
    )


def _validate_model_sampling_rate(
    *,
    trained_sampling_rate_hz: Optional[float],  # noqa: UP045 - Python 3.9/tpcp annotation compatibility.
    sampling_rate_hz: float,
    allow_mismatch: bool,
) -> None:
    if trained_sampling_rate_hz is None or allow_mismatch:
        return
    if not np.isclose(float(trained_sampling_rate_hz), float(sampling_rate_hz)):
        raise ValueError(
            "The XGBoost wear-time model was trained for "
            f"{trained_sampling_rate_hz} Hz, but inference received {sampling_rate_hz} Hz. "
            "Pass a model trained at the target sampling rate or set `allow_sampling_rate_mismatch=True`."
        )


class _FeatureExtractionKwargs(TypedDict):
    sampling_rate_hz: float
    window_samples: int
    step_samples: int
    window_batch_size: int
    version: Literal["full", "lightweight"]
    sensor_cols: tuple[str, ...]
    feature_names: tuple[str, ...]


class _TrainingFeatureKwargs(_FeatureExtractionKwargs):
    memory: Memory
    window_sec: float


def _extract_recording_feature_batches(
    data: pd.DataFrame,
    *,
    sampling_rate_hz: float,
    window_samples: int,
    step_samples: int,
    window_batch_size: int,
    version: Literal["full", "lightweight"],
    sensor_cols: tuple[str, ...],
    feature_names: tuple[str, ...],
) -> list[np.ndarray]:
    window_start_end_ = window_start_end(len(data), window_samples, step_samples)
    dt = 1.0 / sampling_rate_hz
    sensor_data = data.loc[:, list(sensor_cols)].to_numpy(dtype=float, copy=False)
    feature_batches = []
    for batch_start in range(0, len(window_start_end_), window_batch_size):
        batch_start_end = window_start_end_[batch_start : batch_start + window_batch_size]
        feature_batches.append(
            extract_features_batched(
                sensor_data,
                batch_start_end,
                acc_axes=sensor_cols[:3],
                gyr_axes=sensor_cols[3:],
                fs=sampling_rate_hz,
                dt=dt,
                feature_names=feature_names,
                version=version,
            ).to_numpy(dtype=np.float32, copy=False)
        )
    return feature_batches


def _recording_feature_batches(
    data: pd.DataFrame, *, memory: Memory, feature_kwargs: _FeatureExtractionKwargs
) -> list[np.ndarray]:
    extract = _extract_recording_feature_batches
    if memory.location is not None:
        extract = hybrid_cache(memory, False, fast_inaccurate_hashing=True)(extract)
    return extract(data, **feature_kwargs)


def _iter_training_recording_feature_batches(
    data: pd.DataFrame,
    reference_weartime: pd.DataFrame,
    *,
    memory: Memory,
    window_sec: float,
    **feature_kwargs: Unpack[_FeatureExtractionKwargs],
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    window_start_end_ = window_start_end(len(data), feature_kwargs["window_samples"], feature_kwargs["step_samples"])
    feature_batches = _recording_feature_batches(data, memory=memory, feature_kwargs=feature_kwargs)
    reference_centers = window_start_end_[:, 0] + int((window_sec * feature_kwargs["sampling_rate_hz"]) // 2)
    for batch_index, features in enumerate(feature_batches):
        batch_start = batch_index * feature_kwargs["window_batch_size"]
        labels = labels_from_interval_centers(
            reference_centers[batch_start : batch_start + len(features)], reference_weartime
        )
        yield features, labels


def _extract_training_data_recording_features(
    training_data: Any,
    datapoint_index: int,
    **kwargs: Unpack[_TrainingFeatureKwargs],
) -> list[tuple[np.ndarray, np.ndarray]]:
    data, reference_weartime = training_data.load_recording(datapoint_index)
    return list(_iter_training_recording_feature_batches(data, reference_weartime, **kwargs))


def _iter_training_feature_results(
    training_data: TrainingData,
    *,
    n_jobs: int,
    **kwargs: Unpack[_TrainingFeatureKwargs],
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if n_jobs != 1 and hasattr(training_data, "load_recording") and hasattr(training_data, "__len__"):
        recording_batches = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            return_as="generator",
            pre_dispatch="n_jobs",
        )(
            delayed(_extract_training_data_recording_features)(training_data, datapoint_index, **kwargs)
            for datapoint_index in range(len(training_data))
        )
        for recording_batch_results in recording_batches:
            yield from recording_batch_results
    else:
        for data, reference_weartime in training_data:
            yield from _iter_training_recording_feature_batches(data, reference_weartime, **kwargs)


@base_weartime_docfiller
class WtdMegaritisXGBoost(BaseWeartimeDetector):
    """
    XGBoost-based wear-time detection for lower-back worn IMU sensors.

    The detector extracts time-domain and frequency-domain features from overlapping 5-second windows and classifies
    each window with a sklearn-compatible classifier. Packaged XGBoost models are available for the full 230-feature
    set and a 79-feature lightweight set selected from 90%% SHAP importance.

    Parameters
    ----------
    clf
        Fitted or trainable sklearn classifier exposing ``fit`` and binary ``predict_proba`` with classes ``[0, 1]``.
        If ``None``, ``detect`` lazily loads the packaged pretrained model selected by ``version`` and
        ``self_optimize`` creates a new ``xgboost.XGBClassifier`` with the production hyperparameters.
    feature_names
        Feature order expected by ``clf``. If ``None``, the original production order for ``version`` is used.
    version
        Feature/model variant: ``"lightweight"`` for 79 features or ``"full"`` for 230 features.
    window_sec
        Window size in seconds for feature extraction.
    overlap
        Window overlap fraction.
    sensor_cols
        Body-frame sensor columns used for feature extraction.
    prediction_threshold
        Probability threshold used to convert wear probabilities to binary window predictions.
    window_batch_size
        Number of windows feature-extracted before one classifier prediction call.
    n_jobs
        Number of process workers used to extract dataset-backed training datapoints in ``self_optimize``. ``1``
        disables parallel training feature extraction; ``-1`` uses all available workers. Parallelism is only over
        indexed datapoints; generic training iterables without indexed loading still run sequentially.
    memory
        Optional joblib cache for complete recording-level feature batches shared by training and detection. Disabled
        by default. Cache entries use float32 features; feature calculations retain their original precision.
    waking_hours
        Local waking-hours window used for ``total_weartime_during_waking_min_`` as ``(start, end)``.
    trained_sampling_rate_hz
        Sampling rate used to train ``clf``. Pretrained models use 100 Hz.
    allow_sampling_rate_mismatch
        If ``True``, allow inference at a different sampling rate than ``trained_sampling_rate_hz``.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(weartime_list_)s
    %(total_weartime_samples_)s
    %(total_weartime_min_)s
    %(total_weartime_during_waking_min_)s
    clf_
        The classifier used for the latest ``detect`` call.
    feature_matrix_
        Extracted window-level feature matrix for the latest ``detect`` call.
    window_predictions_
        Binary window-level wear predictions.
    window_probabilities_
        Window-level wear probabilities.
    window_start_end_
        ``[start, end)`` sample boundaries for each classified window.
    %(perf_)s
    """

    clf: OptimizableParameter[Optional[_SklearnWearTimeClassifier]]  # noqa: UP045 - tpcp 2.1 resolves annotations.
    feature_names: OptimizableParameter[Optional[Sequence[str]]]  # noqa: UP045 - tpcp 2.1 resolves annotations.
    trained_sampling_rate_hz: OptimizableParameter[Optional[float]]  # noqa: UP045 - tpcp 2.1 resolves annotations.
    window_predictions_: np.ndarray
    window_probabilities_: np.ndarray
    window_start_end_: np.ndarray
    feature_matrix_: pd.DataFrame

    _feature_names_by_version: Final = {
        "full": tuple(FULL_FEATURE_ORDER),
        "lightweight": tuple(FEATURE_ORDER_90PCT),
    }

    class PredefinedParameters:
        """Predefined parameters for the XGBoost wear-time detector."""

        class _ModelConfig(TypedDict):
            clf: _SklearnWearTimeClassifier
            feature_names: tuple[str, ...]
            version: Literal["full", "lightweight"]
            trained_sampling_rate_hz: float

        @classmethod
        def _load_model_config(cls, version: Literal["full", "lightweight"]) -> MappingProxyType[str, Any]:
            if version == "full":
                model_file = "xgboost_fullfeatures_lowback_model.pkl"
                feature_order_file = "xgboost_fullfeatures_lowback_feature_order.pkl"
            else:
                model_file = "xgboost_90pct_lowback_model.pkl"
                feature_order_file = "xgboost_90pct_lowback_feature_order.pkl"

            return MappingProxyType(
                {
                    "clf": _load_pickle_resource(model_file),
                    "feature_names": tuple(_load_pickle_resource(feature_order_file)),
                    "version": version,
                    "trained_sampling_rate_hz": 100.0,
                }
            )

        @classproperty
        def lightweight(cls) -> _ModelConfig:  # noqa: N805
            return cls._load_model_config("lightweight")

        @classproperty
        def full(cls) -> _ModelConfig:  # noqa: N805
            return cls._load_model_config("full")

        @classproperty
        def untrained_lightweight(cls) -> _ModelConfig:  # noqa: N805
            return MappingProxyType(
                {
                    "clf": _new_xgboost_classifier(),
                    "feature_names": tuple(FEATURE_ORDER_90PCT),
                    "version": "lightweight",
                    "trained_sampling_rate_hz": None,
                }
            )

        @classproperty
        def untrained_full(cls) -> _ModelConfig:  # noqa: N805
            return MappingProxyType(
                {
                    "clf": _new_xgboost_classifier(),
                    "feature_names": tuple(FULL_FEATURE_ORDER),
                    "version": "full",
                    "trained_sampling_rate_hz": None,
                }
            )

    class OptimizationPresets:
        """Dataset-specific search settings for an untrained detector in WtdEmulationPipeline."""

        @staticmethod
        def _create_search_space(trial: Any) -> None:
            for name, choices in {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.6, 0.8],
                "min_child_weight": [1, 3],
            }.items():
                trial.suggest_categorical(f"algo__clf__{name}", choices)

        @classproperty
        def sustain_weartime(cls) -> MappingProxyType[str, Any]:  # noqa: N805
            """Human-only three-fold ranking with 40% human and five seeded simulated non-wear training days."""
            return MappingProxyType(
                {
                    "create_search_space": cls._create_search_space,
                    **_sustain_weartime_optimization_defaults(),
                }
            )

    def __init__(
        self,
        *,
        clf: Optional[_SklearnWearTimeClassifier] = None,  # noqa: UP045 - tpcp 2.1 resolves annotations.
        feature_names: Optional[Sequence[str]] = None,  # noqa: UP045 - tpcp 2.1 resolves annotations.
        version: Literal["full", "lightweight"] = "lightweight",
        window_sec: float = 5.0,
        overlap: float = 0.75,
        sensor_cols: Sequence[str] = tuple(BF_SENSOR_COLS),
        prediction_threshold: float = 0.5,
        window_batch_size: int = 4096,
        n_jobs: int = 1,
        memory: Memory = Memory(None),
        waking_hours: tuple[time, time] = (time(7), time(22)),
        trained_sampling_rate_hz: Optional[float] = 100.0,  # noqa: UP045 - tpcp 2.1 resolves annotations.
        allow_sampling_rate_mismatch: bool = False,
    ) -> None:
        self.clf = clf
        self.feature_names = feature_names
        self.version = version
        self.window_sec = window_sec
        self.overlap = overlap
        self.sensor_cols = sensor_cols
        self.prediction_threshold = prediction_threshold
        self.window_batch_size = window_batch_size
        self.n_jobs = n_jobs
        self.memory = memory
        self.waking_hours = waking_hours
        self.trained_sampling_rate_hz = trained_sampling_rate_hz
        self.allow_sampling_rate_mismatch = allow_sampling_rate_mismatch

    @make_action_safe
    @timed_action_method
    @base_weartime_docfiller
    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_: Unpack[dict[str, Any]]) -> Self:
        """%(detect_short)s using an XGBoost classifier with overlapping feature windows."""
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        _validate_waking_hours(self.waking_hours)
        assert_is_sensor_data(data, frame="body")

        clf, feature_names, trained_sampling_rate_hz = self._classifier_and_feature_names()
        check_is_fitted(clf)
        _validate_model_sampling_rate(
            trained_sampling_rate_hz=trained_sampling_rate_hz,
            sampling_rate_hz=sampling_rate_hz,
            allow_mismatch=self.allow_sampling_rate_mismatch,
        )
        self.clf_ = clf

        window_samples, step_samples = self._window_parameters(sampling_rate_hz)
        self.window_start_end_ = window_start_end(len(data), window_samples, step_samples)

        feature_batches: list[pd.DataFrame] = []
        probability_batches: list[np.ndarray] = []
        for features in self._iter_feature_batches(
            data,
            sampling_rate_hz=sampling_rate_hz,
            window_start_end_=self.window_start_end_,
            feature_names=feature_names,
        ):
            feature_batches.append(features)
            probability_batches.append(clf.predict_proba(features)[:, 1].astype(np.float32, copy=False))

        self.feature_matrix_ = (
            pd.concat(feature_batches, axis=0, ignore_index=True)
            if feature_batches
            else pd.DataFrame(columns=feature_names)
        )
        self.window_probabilities_ = (
            np.concatenate(probability_batches).astype(np.float32, copy=False)
            if probability_batches
            else np.empty(0, dtype=np.float32)
        )
        self.window_predictions_ = (self.window_probabilities_ >= self.prediction_threshold).astype(np.int32)

        if len(self.window_predictions_) == 0:
            self.weartime_list_ = _unify_weartime_df(pd.DataFrame(columns=["start", "end"]).rename_axis(index="wt_id"))
            self.weartime_list_ = _with_local_datetimes(self.weartime_list_, data, sampling_rate_hz)
            return self

        weartime_flags, vote_counts = overlapping_window_predictions_to_sample_labels(
            predictions=self.window_predictions_.tolist(),
            data_length=len(data),
            window_samples=window_samples,
            step_samples=step_samples,
        )
        weartime_intervals = bool_array_to_start_end_array(weartime_flags).astype(np.int64, copy=False).reshape(-1, 2)
        weartime_intervals = filter_short_wear_bouts_by_confidence(
            wear_intervals=weartime_intervals,
            vote_counts=vote_counts,
            data_length=len(data),
            sampling_rate_hz=sampling_rate_hz,
            min_confidence_short_bouts=0.90,
            short_bout_threshold_minutes=20,
            min_bout_duration_seconds=15,
        )
        weartime_intervals = remove_isolated_short_periods_from_intervals(
            weartime_intervals,
            data_length=len(data),
            min_period_s=15,
            sampling_rate_hz=sampling_rate_hz,
        )
        self.weartime_list_ = pd.DataFrame(weartime_intervals, columns=["start", "end"]).rename_axis(index="wt_id")
        self.weartime_list_["end"] = self.weartime_list_["end"].clip(upper=len(data))
        self.weartime_list_ = _unify_weartime_df(self.weartime_list_)
        self.weartime_list_ = _with_local_datetimes(self.weartime_list_, data, sampling_rate_hz)

        return self

    @make_optimize_safe
    @base_weartime_docfiller
    def self_optimize(
        self,
        training_data: TrainingData,
        *,
        sampling_rate_hz: float,
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        """Fit the configured classifier from lazy recording-level training data."""
        clf = self.clf or _new_xgboost_classifier()

        window_samples, step_samples = self._window_parameters(sampling_rate_hz)
        if self.window_batch_size <= 0:
            raise ValueError("`window_batch_size` must be a positive integer.")
        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError("`n_jobs` must be -1 or a positive integer.")

        feature_names = tuple(self.feature_names or self._feature_names_by_version[self.version])
        sensor_cols = tuple(self.sensor_cols)
        feature_batches = []
        label_batches = []

        for recording_features, recording_labels in _iter_training_feature_results(
            training_data,
            memory=self.memory,
            n_jobs=self.n_jobs,
            sampling_rate_hz=sampling_rate_hz,
            window_samples=window_samples,
            step_samples=step_samples,
            window_sec=self.window_sec,
            window_batch_size=self.window_batch_size,
            version=self.version,
            sensor_cols=sensor_cols,
            feature_names=feature_names,
        ):
            feature_batches.append(recording_features)
            label_batches.append(recording_labels)

        if not feature_batches or not any(len(batch) for batch in feature_batches):
            raise ValueError("The training data did not yield any XGBoost feature windows.")

        feature_values = feature_batches[0] if len(feature_batches) == 1 else np.concatenate(feature_batches)
        all_labels = label_batches[0] if len(label_batches) == 1 else np.concatenate(label_batches)
        all_features = pd.DataFrame(feature_values, columns=feature_names, copy=False)
        clf.fit(all_features, all_labels, **kwargs)

        self.clf = clf
        self.feature_names = tuple(all_features.columns)
        self.trained_sampling_rate_hz = float(sampling_rate_hz)
        return self

    def extract_features(self, data: pd.DataFrame, sampling_rate_hz: float) -> pd.DataFrame:
        """Extract the configured window-level feature matrix without running the classifier."""
        assert_is_sensor_data(data, frame="body")
        window_samples, step_samples = self._window_parameters(sampling_rate_hz)
        window_start_end_ = window_start_end(len(data), window_samples, step_samples)
        feature_names = tuple(self.feature_names or self._feature_names_by_version[self.version])
        feature_batches = list(
            self._iter_feature_batches(
                data,
                sampling_rate_hz=sampling_rate_hz,
                window_start_end_=window_start_end_,
                feature_names=feature_names,
            )
        )
        return (
            pd.concat(feature_batches, axis=0, ignore_index=True)
            if feature_batches
            else pd.DataFrame(columns=feature_names)
        )

    def _classifier_and_feature_names(
        self,
    ) -> tuple[_SklearnWearTimeClassifier, tuple[str, ...], Optional[float]]:  # noqa: UP045
        if self.clf is not None:
            return (
                self.clf,
                tuple(self.feature_names or self._feature_names_by_version[self.version]),
                self.trained_sampling_rate_hz,
            )

        model_config = (
            self.PredefinedParameters.full if self.version == "full" else self.PredefinedParameters.lightweight
        )
        return model_config["clf"], model_config["feature_names"], model_config["trained_sampling_rate_hz"]

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

    def _iter_feature_batches(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        window_start_end_: np.ndarray,
        feature_names: Sequence[str],
    ) -> Iterator[pd.DataFrame]:
        if len(window_start_end_) == 0:
            return
        if self.window_batch_size <= 0:
            raise ValueError("`window_batch_size` must be a positive integer.")

        window_samples, step_samples = self._window_parameters(sampling_rate_hz)
        feature_kwargs: _FeatureExtractionKwargs = {
            "sampling_rate_hz": sampling_rate_hz,
            "window_samples": window_samples,
            "step_samples": step_samples,
            "window_batch_size": self.window_batch_size,
            "version": self.version,
            "sensor_cols": tuple(self.sensor_cols),
            "feature_names": tuple(feature_names),
        }
        for feature_values in _recording_feature_batches(data, memory=self.memory, feature_kwargs=feature_kwargs):
            yield pd.DataFrame(feature_values, columns=feature_names)
