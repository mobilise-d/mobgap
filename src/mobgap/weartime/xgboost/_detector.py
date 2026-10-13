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

import json
import pickle
from collections.abc import Iterator, Sequence  # noqa: TC003 - tpcp resolves algorithm annotations at runtime.
from dataclasses import replace
from datetime import time
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd
from joblib import Memory, parallel_config
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from tpcp import cf, clone, make_action_safe
from tpcp.caching import hybrid_cache
from tpcp.misc import classproperty, set_defaults
from tpcp.parallel import Parallel, delayed
from typing_extensions import Self, TypedDict, Unpack
from xgboost import XGBClassifier

from mobgap._utils_internal.misc import timed_action_method
from mobgap.consts import BF_SENSOR_COLS
from mobgap.utils.array_handling import bool_array_to_start_end_array
from mobgap.utils.dtypes import assert_is_sensor_data
from mobgap.weartime._model_config import WearTimeModelConfig
from mobgap.weartime.base import (
    BaseWeartimeDetector,
    _unify_weartime_df,
    base_weartime_docfiller,
)
from mobgap.weartime.utils._intervals import _validate_waking_hours, _with_local_datetimes
from mobgap.weartime.utils.ml_feature_extraction import window_start_end
from mobgap.weartime.utils.windows_to_weartime import (
    filter_short_wear_bouts_by_confidence,
    overlapping_window_predictions_to_sample_labels,
    remove_isolated_short_periods_from_intervals,
)
from mobgap.weartime.xgboost._features import (
    FEATURE_ORDER_90PCT,
    _prepare_lightweight_recording,
    extract_features_batched,
)

_SklearnWearTimeClassifier = BaseEstimator


@lru_cache(maxsize=None)  # noqa: UP033 - Use the Python 3.8-compatible spelling.
def _load_pickle_resource(file_name: str) -> Any:
    resource = files("mobgap.weartime.xgboost._models").joinpath(file_name)
    with resource.open("rb") as file:
        return pickle.load(file)


class _FeatureExtractionKwargs(TypedDict):
    sampling_rate_hz: float
    window_samples: int
    step_samples: int
    window_batch_size: int
    version: Literal["full", "lightweight"]
    sensor_cols: tuple[str, ...]
    feature_names: tuple[str, ...]


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
    """Extract float32 batches with shared signals, compensated moments and wide quartile interpolation."""
    window_start_end_ = window_start_end(len(data), window_samples, step_samples)
    if len(window_start_end_) == 0:
        return []
    dt = 1.0 / sampling_rate_hz
    # Named features use anatomical axes, independently of the configured input order.
    sensor_frame = data.loc[:, sorted(sensor_cols, key=BF_SENSOR_COLS.index)]
    sensor_data = sensor_frame.to_numpy(copy=False)
    if sensor_data.dtype not in (np.float32, np.float64):
        sensor_data = sensor_frame.to_numpy(dtype=float, copy=False, na_value=np.nan)
    prepared = (
        _prepare_lightweight_recording(sensor_data, window_samples, step_samples, dt)
        if version == "lightweight"
        else None
    )

    def extract_batch(batch_start_end: np.ndarray) -> np.ndarray:
        return extract_features_batched(
            sensor_data,
            batch_start_end,
            fs=sampling_rate_hz,
            dt=dt,
            feature_names=feature_names,
            version=version,
            _prepared=prepared,
        ).to_numpy(dtype=np.float32, copy=False)

    return Parallel()(
        delayed(extract_batch)(window_start_end_[start : start + window_batch_size])
        for start in range(0, len(window_start_end_), window_batch_size)
    )


def _recording_feature_batches(
    data: pd.DataFrame, *, memory: Memory, feature_kwargs: _FeatureExtractionKwargs, n_jobs: int = 1
) -> list[np.ndarray]:
    extract = _extract_recording_feature_batches
    if memory.location is not None:
        extract = hybrid_cache(memory, False, fast_inaccurate_hashing=True)(extract)
    # Execution policy stays outside the cached arguments: training and inference
    # share features even when they use different numbers of workers.
    with parallel_config(backend="threading", n_jobs=n_jobs):
        return extract(data, **feature_kwargs)


@base_weartime_docfiller
class WtdMegaritisXGBoost(BaseWeartimeDetector):
    """
    XGBoost-based wear-time detection for lower-back worn IMU sensors.

    The detector extracts time-domain and frequency-domain features from overlapping windows and classifies
    each window with a sklearn-compatible classifier. The full extractor computes 230 features; the lightweight
    extractor uses 79 features selected from 90%% SHAP importance. Supply a trained classifier for detection.
    ``PredefinedParameters.sustain_weartime`` loads the packaged fitted classifier and its input configuration.

    Parameters
    ----------
    clf
        Sklearn-compatible classifier defining the model type and parameters. Supply it untrained to the Trainer;
        detection requires fitted state and binary predict_proba with classes [0, 1].
    model_config
        Window duration, channel order and feature order owned by this detector. An unknown sampling rate is filled
        when fitting; the completed configuration carries the native rate required for inference.
    overlap
        Fractional overlap of inference windows. Training overlap is configured on the trainer.
    prediction_threshold
        Wear probabilities greater than or equal to this value become wear predictions.
    window_batch_size
        Number of feature windows processed in one batch.
    n_jobs_features
        Number of joblib threads over feature batches within a recording.
    memory
        Optional joblib cache for complete recording feature batches. Worker count does not change cache keys.
    waking_hours
        Local waking-hours interval used for the reported waking wear time.

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
    window_predictions_
        Binary window-level wear predictions.
    window_probabilities_
        Window-level wear probabilities.
    window_start_end_
        ``[start, end)`` sample boundaries for each classified window.
    %(perf_)s
    """

    clf: _SklearnWearTimeClassifier
    model_config: WearTimeModelConfig
    window_predictions_: np.ndarray
    window_probabilities_: np.ndarray
    window_start_end_: np.ndarray

    class PredefinedParameters:
        """Fitted SUSTAIN model artifacts and their native input configuration."""

        @classproperty
        def empty_xgboost(cls) -> MappingProxyType[str, Any]:  # noqa: N805
            """Create a fresh XGBoost classifier with baseline settings and no fitted state."""
            return MappingProxyType(
                {
                    "clf": XGBClassifier(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_weight=3,
                        random_state=42,
                        n_jobs=4,
                    ),
                    "model_config": WearTimeModelConfig(feature_names=tuple(FEATURE_ORDER_90PCT)),
                    "overlap": 0.75,
                    "prediction_threshold": 0.5,
                    "window_batch_size": 4096,
                    "n_jobs_features": 4,
                    "memory": Memory(None),
                    "waking_hours": (time(7), time(22)),
                }
            )

        @classproperty
        def sustain_weartime(cls) -> MappingProxyType[str, Any]:  # noqa: N805
            """Load the fitted classifier and its exact trained input requirements."""
            with (
                files("mobgap.weartime.xgboost._models")
                .joinpath("xgboost_sustain_weartime_model_config.json")
                .open() as file
            ):
                config = WearTimeModelConfig.from_dict(json.load(file))
            return MappingProxyType(
                {"clf": clone(_load_pickle_resource("xgboost_sustain_weartime_model.pkl")), "model_config": config}
            )

    @set_defaults(**{k: cf(v) for k, v in PredefinedParameters.empty_xgboost.items()})
    def __init__(
        self,
        *,
        clf: _SklearnWearTimeClassifier,
        model_config: WearTimeModelConfig,
        overlap: float,
        prediction_threshold: float,
        window_batch_size: int,
        n_jobs_features: int,
        memory: Memory,
        waking_hours: tuple[time, time],
    ) -> None:
        self.clf = clf
        self.model_config = model_config
        self.overlap = overlap
        self.prediction_threshold = prediction_threshold
        self.window_batch_size = window_batch_size
        self.n_jobs_features = n_jobs_features
        self.memory = memory
        self.waking_hours = waking_hours

    @make_action_safe
    @timed_action_method
    @base_weartime_docfiller
    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_: Unpack[dict[str, Any]]) -> Self:
        """%(detect_short)s using an XGBoost classifier with overlapping feature windows."""
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        _validate_waking_hours(self.waking_hours)
        assert_is_sensor_data(data, frame="body")

        clf = self.clf
        check_is_fitted(clf)
        self.model_config.validate_sampling_rate(sampling_rate_hz)
        feature_names = self.model_config.feature_names
        self.clf_ = clf

        window_samples, step_samples = self._window_parameters(sampling_rate_hz)
        self.window_start_end_ = window_start_end(len(data), window_samples, step_samples)

        probability_batches: list[np.ndarray] = []
        for features in self._iter_feature_batches(
            data,
            sampling_rate_hz=sampling_rate_hz,
            window_start_end_=self.window_start_end_,
            feature_names=feature_names,
        ):
            probability_batches.append(clf.predict_proba(features)[:, 1].astype(np.float32, copy=False))

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

    def extract_features(self, data: pd.DataFrame, sampling_rate_hz: float) -> pd.DataFrame:
        """Extract features without running the classifier, including from an untrained input configuration."""
        assert_is_sensor_data(data, frame="body")
        window_samples, step_samples = self._window_parameters(sampling_rate_hz)
        window_start_end_ = window_start_end(len(data), window_samples, step_samples)
        feature_names = self.model_config.feature_names
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

    def _window_parameters(self, sampling_rate_hz: float) -> tuple[int, int]:
        config = self.model_config
        if config.sampling_rate_hz is None:
            config = replace(config, sampling_rate_hz=sampling_rate_hz)
        config.validate_sampling_rate(sampling_rate_hz)
        return config.window_parameters(self.overlap)

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
            "sampling_rate_hz": self.model_config.sampling_rate_hz
            if self.model_config.sampling_rate_hz is not None
            else sampling_rate_hz,
            "window_samples": window_samples,
            "step_samples": step_samples,
            "window_batch_size": self.window_batch_size,
            "version": "lightweight" if set(feature_names).issubset(FEATURE_ORDER_90PCT) else "full",
            "sensor_cols": self.model_config.sensor_cols,
            "feature_names": tuple(feature_names),
        }
        for feature_values in _recording_feature_batches(
            data, memory=self.memory, feature_kwargs=feature_kwargs, n_jobs=self.n_jobs_features
        ):
            yield pd.DataFrame(feature_values, columns=feature_names)
