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

from contextlib import suppress
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

import pandas as pd
from tpcp import OptimizableParameter, make_action_safe, make_optimize_safe
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import timed_action_method
from mobgap.weartime.base import (
    BaseWeartimeDetector,
    RecordingSampleCounts,
    TrainingData,
    _unify_weartime_df,
    base_weartime_docfiller,
)
from mobgap.weartime.utils._intervals import _validate_waking_hours_min, flags_to_intervals
from mobgap.weartime.utils.windows_to_weartime import (
    filter_short_wear_bouts_by_confidence,
    overlapping_window_predictions_to_sample_labels,
    remove_isolated_short_periods_from_intervals,
)

if TYPE_CHECKING:
    from mobgap.weartime._keras_weartime_model import BaseKerasWeartimeModel

_C = TypeVar("_C", bound=Callable[..., Any])


def _make_action_safe(action_method: _C) -> _C:
    """Apply tpcp action checks while staying compatible with tpcp 2.1's test mixin."""
    safe_action_method = make_action_safe(action_method)
    with suppress(AttributeError):
        delattr(safe_action_method, "__tpcp_action_method")
    return safe_action_method


@base_weartime_docfiller
class WtdMegaritisCNN(BaseWeartimeDetector):
    """
    1D CNN-based wear-time detection for lower-back worn IMU sensors.

    Uses a pre-trained 1D Convolutional Neural Network trained on raw windowed
    IMU data (accelerometer and gyroscope). Processes overlapping 5-second windows
    with per-window scaling and includes biomechanically-informed post-processing.

    Post-processing steps:
    1. Majority voting across overlapping windows to obtain sample-level predictions
    2. Removal of wear bouts shorter than 15 seconds (biomechanically implausible)
    3. Confidence filtering for wear bouts under 20 minutes (requires >90%% vote agreement)
    4. Merging of short non-wear gaps (<15s) between wear periods

    Parameters
    ----------
    model : BaseKerasWeartimeModel, optional
        Low-level Keras window classifier. Pass a pretrained or trainable model instance explicitly before calling
        ``detect`` or ``self_optimize``.
    waking_hours_min : tuple[int, int]
        Waking-hours window used for ``total_weartime_during_waking_min_`` as ``(start, end)`` in minutes since
        midnight.

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
    In addition to total wear-time, this algorithm calculates wear-time during waking hours
    (07:00-22:00 by default), required for Mobilise-D DMO weekly aggregation. The waking hours value is
    extracted from the post-processed sample-level predictions by filtering wear-time to the
    configured waking-hours window.

    The pipeline is designed for daily recordings (midnight-to-midnight, ~24 hours).
    For recordings shorter than the configured waking-hours end, the algorithm issues a warning and uses
    ``total_weartime_min_`` as a fallback for ``total_weartime_during_waking_min_``. For recordings longer than
    one day, accessing ``total_weartime_during_waking_min_`` raises an error because the recording must be segmented
    per day before a single daily waking-hours window can be applied.
    Waking hours are identified using sample indices (07:00 = 7x3600xsampling_rate_hz) rather than
    timestamps, ensuring compatibility with devices that may not provide timestamp metadata.

    **Model Architecture**
    The low-level model operates on raw windowed IMU data with per-window standardization
    (features scaled to zero mean, unit variance per window). Packaged production models can be instantiated with the
    ``PredefinedParameters`` of the respective low-level model class and then passed as ``model``.

    Model architecture: 3-layer 1D CNN with progressively increasing filters [32, 64, 128],
    kernel size 9, max pooling (size 2), batch normalization, and dropout (0.3). Fully
    connected layer with 64 units. Trained with Adam optimizer (learning rate 0.001,
    batch size 1024). CNN-LSTM variant includes 64-unit LSTM layer before dense layer.
    """

    model: OptimizableParameter[Optional[Any]]  # noqa: UP045 - tpcp 2.1 needs Python 3.9-evaluable strings.

    def __init__(
        self,
        *,
        model: BaseKerasWeartimeModel | None = None,
        waking_hours_min: tuple[int, int] = (7 * 60, 22 * 60),
    ) -> None:
        self.model = model
        self.waking_hours_min = waking_hours_min

    @_make_action_safe
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

        Processes raw IMU data in overlapping windows with per-window standardization,
        applies the pre-trained CNN model, and converts window-level predictions to
        sample-level wear-time segments using majority voting and biomechanical
        post-processing rules.

        Parameters
        ----------
        %(detect_para)s

        %(detect_return)s

        Notes
        -----
        Each window is independently standardized (zero mean, unit variance) before
        being fed to the CNN, matching the training preprocessing.
        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        data_length = len(data)
        _validate_waking_hours_min(self.waking_hours_min)

        model = self.model
        if model is None:
            raise RuntimeError("Pass a Keras wear-time `model` before calling `detect`.")
        self.model_ = model.clone().run(data, sampling_rate_hz=sampling_rate_hz)

        # Post-processing: convert window predictions to sample-level weartime
        weartime_flags, vote_counts = overlapping_window_predictions_to_sample_labels(
            predictions=self.model_.window_predictions_.tolist(),
            data_length=data_length,
            window_samples=self.model_.window_samples_,
            step_samples=self.model_.step_samples_,
        )
        weartime_intervals = flags_to_intervals(weartime_flags)
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
            min_period_sec=15,
            sampling_rate_hz=sampling_rate_hz,
        )
        self.weartime_list_ = pd.DataFrame(weartime_intervals, columns=["start", "end"]).rename_axis(index="wt_id")

        # Ensure end indices don't exceed data length
        self.weartime_list_["end"] = self.weartime_list_["end"].clip(upper=data_length)

        # Unify format (adds wt_id index, ensures correct dtypes)
        self.weartime_list_ = _unify_weartime_df(self.weartime_list_)

        return self

    @make_optimize_safe
    def self_optimize(
        self,
        training_data: TrainingData,
        *,
        sampling_rate_hz: float,
        recording_sample_counts: RecordingSampleCounts,
    ) -> Self:
        """Train the low-level Keras window model from lazy ``(data, reference_weartime)`` records."""
        model = self.model
        if model is None:
            raise RuntimeError("Pass an untrained Keras wear-time `model` before calling `self_optimize`.")
        self.model = model.self_optimize(
            training_data,
            sampling_rate_hz=sampling_rate_hz,
            recording_sample_counts=recording_sample_counts,
        )
        return self
