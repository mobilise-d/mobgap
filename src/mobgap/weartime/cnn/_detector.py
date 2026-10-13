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
from datetime import time
from importlib.resources import files
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from tpcp import cf, clone, make_action_safe
from tpcp.misc import classproperty, set_defaults
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import timed_action_method
from mobgap.utils.array_handling import bool_array_to_start_end_array
from mobgap.weartime._model_config import WearTimeModelConfig
from mobgap.weartime.base import (
    BaseWeartimeDetector,
    _unify_weartime_df,
    base_weartime_docfiller,
)
from mobgap.weartime.cnn._model import (
    KerasWeartimeModel,
    MegaritisCnnLstmWeartimeModel,
    MegaritisCnnWeartimeModel,
    _load_keras_model_resource,
)
from mobgap.weartime.utils._intervals import _validate_waking_hours, _with_local_datetimes
from mobgap.weartime.utils.windows_to_weartime import (
    filter_short_wear_bouts_by_confidence,
    overlapping_window_predictions_to_sample_labels,
    remove_isolated_short_periods_from_intervals,
)


def _load_cnn_parameters(prefix: str, model_type: type[KerasWeartimeModel]) -> MappingProxyType[str, Any]:
    resources = files("mobgap.weartime.cnn._models")
    with resources.joinpath(f"{prefix}_model_config.json").open() as file:
        config = WearTimeModelConfig.from_dict(json.load(file))
    with resources.joinpath(f"{prefix}_model_params.json").open() as file:
        model_params = json.load(file)
    model = clone(_load_keras_model_resource(f"{prefix}_model.keras"))
    return MappingProxyType({"model": model_type(model=model, model_config=config, **model_params)})


@base_weartime_docfiller
class WtdMegaritisCNN(BaseWeartimeDetector):
    """
    1D CNN-based wear-time detection for lower-back worn IMU sensors.

    Uses a pre-trained 1D Convolutional Neural Network trained on raw windowed
    IMU data (accelerometer and gyroscope). Processes overlapping windows at the configured
    duration. Built-in CNN and CNN-LSTM graphs standardize each window internally; supplied
    native graphs receive raw windows and must contain their required preprocessing.

    Post-processing steps:

    1. Majority voting across overlapping windows to obtain sample-level predictions
    2. Removal of short wear bouts
    3. Confidence filtering for short interior wear bouts using wear vote agreement. Recording-boundary bouts are exempt from this confidence filter.
    4. Merging of short non-wear gaps between wear periods

    These duration and confidence thresholds are empirical post-processing heuristics.

    Parameters
    ----------
    model : KerasWeartimeModel
        Fitted low-level window classifier carrying the native model input configuration.
        Use WtdMegaritisCNNTrainer to create and fit a new network.
    waking_hours : tuple[datetime.time, datetime.time]
        Local waking-hours window used for ``total_weartime_during_waking_min_`` as ``(start, end)``.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(weartime_list_)s
    %(total_weartime_samples_)s
    %(total_weartime_min_)s
    %(total_weartime_during_waking_min_)s
    model_
        The low-level Keras wear-time model instance after window-level prediction.
    %(perf_)s

    Notes
    -----
    **Waking Hours Calculation**
    In addition to total wear-time, this algorithm calculates wear-time during the configured waking hours,
    required for Mobilise-D DMO weekly aggregation. The waking hours value is
    extracted from the post-processed sample-level predictions by filtering wear-time to the
    configured waking-hours window.

    Recordings must be segmented per day. A timezone-aware ``DatetimeIndex`` defines the local waking-hours window.
    Otherwise, sample zero is assumed to be midnight with a warning, and DST cannot be considered.

    **Model Architecture**
    The low-level model passes raw windowed IMU data to the native graph. Built-in Megaritis graphs
    standardize each channel to zero mean and unit variance per window; supplied graphs own their preprocessing.
    Supply a trained low-level model for detection.
    The independent SUSTAIN CNN and CNN-LSTM presets load their model definitions, native inputs and fitted weights.
    ``PredefinedParameters.empty_cnn`` and ``empty_cnn_lstm`` create fresh model blueprints
    without loading any packaged artifacts.

    Built-in graphs combine convolution, pooling, batch normalization and dropout before a dense output.
    The CNN-LSTM variant adds a recurrent layer. The model blueprint owns the architecture settings;
    the Trainer owns the Adam optimizer and fitting settings.
    """

    model: KerasWeartimeModel

    class PredefinedParameters:
        """Fresh blueprints and independent SUSTAIN model definitions for CNN and CNN-LSTM."""

        @classproperty
        def empty_cnn(cls) -> MappingProxyType[str, Any]:  # noqa: N805
            """Create a fresh CNN blueprint with default structure and no fitted weights."""
            return MappingProxyType({"model": MegaritisCnnWeartimeModel(), "waking_hours": (time(7), time(22))})

        @classproperty
        def empty_cnn_lstm(cls) -> MappingProxyType[str, Any]:  # noqa: N805
            """Create a fresh CNN-LSTM blueprint with default structure and no fitted weights."""
            return MappingProxyType({**cls.empty_cnn, "model": MegaritisCnnLstmWeartimeModel()})

        @classproperty
        def sustain_weartime_cnn(cls) -> MappingProxyType[str, Any]:  # noqa: N805
            """Load the fitted CNN and its complete model definition."""
            return _load_cnn_parameters("cnn_sustain_weartime", MegaritisCnnWeartimeModel)

        @classproperty
        def sustain_weartime_cnn_lstm(cls) -> MappingProxyType[str, Any]:  # noqa: N805
            """Load the fitted CNN-LSTM and its complete model definition."""
            return _load_cnn_parameters("cnn_lstm_sustain_weartime", MegaritisCnnLstmWeartimeModel)

    @set_defaults(**{k: cf(v) for k, v in PredefinedParameters.empty_cnn.items()})
    def __init__(
        self,
        *,
        model: KerasWeartimeModel,
        waking_hours: tuple[time, time],
    ) -> None:
        self.model = model
        self.waking_hours = waking_hours

    @make_action_safe
    @timed_action_method
    @base_weartime_docfiller
    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """%(detect_short)s using 1D CNN with overlapping windows.

        Passes raw IMU windows to the trained native graph and converts window-level
        predictions to sample-level wear-time segments using majority voting and
        empirical post-processing rules.

        Parameters
        ----------
        %(detect_para)s

        %(detect_return)s

        Notes
        -----
        Built-in CNN and CNN-LSTM graphs perform per-window standardization internally.
        Supplied native graphs receive raw windows and must implement their required preprocessing.
        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        data_length = len(data)
        _validate_waking_hours(self.waking_hours)

        self.model_ = self.model.clone().run(data, sampling_rate_hz=sampling_rate_hz)

        # Post-processing: convert window predictions to sample-level weartime
        weartime_flags, vote_counts = overlapping_window_predictions_to_sample_labels(
            predictions=self.model_.window_predictions_.tolist(),
            data_length=data_length,
            window_samples=self.model_.window_samples_,
            step_samples=self.model_.step_samples_,
        )
        weartime_intervals = bool_array_to_start_end_array(weartime_flags).astype(np.int64, copy=False).reshape(-1, 2)
        weartime_intervals = filter_short_wear_bouts_by_confidence(
            wear_intervals=weartime_intervals,
            vote_counts=vote_counts,
            data_length=data_length,
            sampling_rate_hz=sampling_rate_hz,
            min_confidence_short_bouts=0.90,
            short_bout_threshold_minutes=20,
            min_bout_duration_seconds=15,
        )
        weartime_intervals = remove_isolated_short_periods_from_intervals(
            weartime_intervals,
            data_length=data_length,
            min_period_s=15,
            sampling_rate_hz=sampling_rate_hz,
        )
        self.weartime_list_ = pd.DataFrame(weartime_intervals, columns=["start", "end"]).rename_axis(index="wt_id")

        # Ensure end indices don't exceed data length
        self.weartime_list_["end"] = self.weartime_list_["end"].clip(upper=data_length)

        # Unify format (adds wt_id index, ensures correct dtypes)
        self.weartime_list_ = _unify_weartime_df(self.weartime_list_)
        self.weartime_list_ = _with_local_datetimes(self.weartime_list_, data, sampling_rate_hz)

        return self
