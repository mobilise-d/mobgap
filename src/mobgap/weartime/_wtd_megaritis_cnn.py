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

from datetime import time
from types import MappingProxyType
from typing import Any, Optional

import numpy as np
import pandas as pd
from tpcp import OptimizableParameter, make_action_safe, make_optimize_safe
from tpcp.misc import classproperty
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import timed_action_method
from mobgap.utils.array_handling import bool_array_to_start_end_array
from mobgap.weartime._keras_weartime_model import BaseKerasWeartimeModel, MegaritisCnnWeartimeModel
from mobgap.weartime._optimization_presets import _sustain_weartime_optimization_defaults
from mobgap.weartime.base import (
    BaseWeartimeDetector,
    TrainingData,
    _unify_weartime_df,
    base_weartime_docfiller,
)
from mobgap.weartime.utils._intervals import _validate_waking_hours, _with_local_datetimes
from mobgap.weartime.utils.windows_to_weartime import (
    filter_short_wear_bouts_by_confidence,
    overlapping_window_predictions_to_sample_labels,
    remove_isolated_short_periods_from_intervals,
)


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
        Low-level Keras window classifier. If ``None``, ``detect`` loads the packaged pretrained CNN and
        ``self_optimize`` creates a fresh, untrained CNN. Pass a model explicitly to use another architecture or
        training configuration.
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
    In addition to total wear-time, this algorithm calculates wear-time during waking hours
    (07:00-22:00 by default), required for Mobilise-D DMO weekly aggregation. The waking hours value is
    extracted from the post-processed sample-level predictions by filtering wear-time to the
    configured waking-hours window.

    Recordings must be segmented per day. A timezone-aware ``DatetimeIndex`` defines the local waking-hours window.
    Otherwise, sample zero is assumed to be midnight with a warning, and DST cannot be considered.

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

    class OptimizationPresets:
        """Dataset-specific search settings for an explicit untrained CNN model in WtdEmulationPipeline."""

        @staticmethod
        def _create_search_space(trial: Any) -> None:
            trial.suggest_float("algo__model__learning_rate", 1e-4, 1e-2, log=True)
            trial.suggest_categorical("algo__model__dropout_rate", [0.2, 0.3, 0.5])
            trial.suggest_categorical("algo__model__batch_size", [256, 512, 1024])

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
        model: BaseKerasWeartimeModel | None = None,
        waking_hours: tuple[time, time] = (time(7), time(22)),
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
        _validate_waking_hours(self.waking_hours)

        model = self.model or MegaritisCnnWeartimeModel(**MegaritisCnnWeartimeModel.PredefinedParameters.lowback)
        self.model_ = model.clone().run(data, sampling_rate_hz=sampling_rate_hz)

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

    @make_optimize_safe
    def self_optimize(
        self,
        training_data: TrainingData,
        *,
        sampling_rate_hz: float,
    ) -> Self:
        """Train the Keras model from lazy labeled-reference pairs.

        Records are ``(data, reference_weartime)`` pairs. The reference has a mandatory unordered categorical
        ``label`` column with categories ``["wear", "uncertain"]``. Uncertain means wear versus non-wear could not
        be determined; the unlabeled complement is known non-wear.
        References use half-open sample boundaries; windows intersecting uncertain intervals are excluded,
        retaining known windows on their original recording grid.
        """
        model = self.model or MegaritisCnnWeartimeModel(standardize_in_model=True)
        self.model = model.self_optimize(
            training_data,
            sampling_rate_hz=sampling_rate_hz,
        )
        return self
