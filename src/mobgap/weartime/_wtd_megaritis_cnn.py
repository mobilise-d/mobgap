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

from functools import cache
from importlib.resources import files
from typing import Any, Literal

import numpy as np
import pandas as pd
from tensorflow import keras
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import timed_action_method
from mobgap.utils.array_handling import sliding_window_view
from mobgap.weartime.base import BaseWeartimeDetector, _unify_weartime_df, base_weartime_docfiller
from mobgap.weartime.utils.windows_to_weartime import overlapping_windows_to_sample_labels


@cache
def _load_cnn_model(version: Literal["cnn", "cnn_lstm"]) -> Any:
    """Load pre-trained CNN model from package resources."""
    if version == "cnn":
        model_file = files("mobgap.weartime.production_models").joinpath("cnn_lowback_model.keras")
    else:  # cnn_lstm
        model_file = files("mobgap.weartime.production_models").joinpath("cnn_lstm_lowback_model.keras")

    return keras.models.load_model(model_file)


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
    window_sec : float
        Window size in seconds (default: 5.0, model trained on 5s windows)
    overlap : float
        Window overlap fraction, 0.0 to 1.0 (default: 0.75)
    version : Literal["cnn", "cnn_lstm"]
        Model variant: "cnn" (baseline) or "cnn_lstm" (with LSTM, default)
    position : Literal['lowback']
        Sensor position (default: 'lowback', only supported position)

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(weartime_list_)s
    %(total_weartime_samples_)s
    %(total_weartime_minutes_)s
    %(total_weartime_hours_)s
    %(total_weartime_hours_during_waking_)s
    %(perf_)s

    Notes
    -----
    **Waking Hours Calculation**
    In addition to total wear-time, this algorithm calculates wear-time during waking hours
    (07:00-22:00), required for Mobilise-D DMO weekly aggregation. The waking hours value is extracted
    from the post-processed sample-level predictions by filtering wear-time to the 07:00-22:00 window.

    The pipeline is designed for daily recordings (midnight-to-midnight, ~24 hours).
    For recordings shorter than 22 hours or longer than 25 hours, the algorithm issues a warning
    and uses ``total_weartime_hours_`` as a fallback for ``total_weartime_hours_during_waking_``,
    as the waking hours window cannot be reliably identified in non-standard recording durations.
    Waking hours are identified using sample indices (07:00 = 7x3600xsampling_rate_hz) rather than
    timestamps, ensuring compatibility with devices that may not provide timestamp metadata.

    **Model Architecture**
    Pre-trained model is loaded from the package's production_models folder.
    CNN operates on raw windowed IMU data with per-window standardization
    (features scaled to zero mean, unit variance per window).

    Model architecture: 3-layer 1D CNN with progressively increasing filters [32, 64, 128],
    kernel size 9, max pooling (size 2), batch normalization, and dropout (0.3). Fully
    connected layer with 64 units. Trained with Adam optimizer (learning rate 0.001,
    batch size 1024). CNN-LSTM variant includes 64-unit LSTM layer before dense layer.

    Pre-trained model is loaded from the package's production_models folder.
    """

    # Type hints
    total_weartime_hours_during_waking_: float

    def __init__(
        self,
        *,
        window_sec: float = 5.0,
        overlap: float = 0.75,
        version: Literal["cnn", "cnn_lstm"] = "cnn_lstm",
        position: Literal["lowback"] = "lowback",
    ) -> None:
        self.window_sec = window_sec
        self.overlap = overlap
        self.version = version
        self.position = position

    @property
    def model(self) -> Any:
        """Lazy-load the model when first accessed."""
        return _load_cnn_model(self.version)

    def __getstate__(self) -> dict:
        """Exclude model from pickling/hashing."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore model after unpickling."""
        self.__dict__.update(state)

    @timed_action_method
    @base_weartime_docfiller
    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float = 100,
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

        win_samples = int(self.window_sec * self.sampling_rate_hz)
        overlap_samples = int(win_samples * self.overlap)

        # Required columns for CNN
        required_cols = ["acc_is", "acc_ml", "acc_pa", "gyr_is", "gyr_ml", "gyr_pa"]

        # Extract sensor data as numpy array
        sensor_data = self.data[required_cols].to_numpy().astype(np.float32)

        # Create sliding windows
        # Output shape: (n_windows, window_size, 6)
        windowed_data = sliding_window_view(sensor_data, win_samples, overlap_samples)

        # Batch standardize all windows
        all_windows = []
        for window in windowed_data:
            # Per-window standardization
            x_mean = window.mean(axis=0)
            x_std = window.std(axis=0)
            x_std[x_std < 1e-8] = 1e-8
            normalized = (window - x_mean) / x_std
            all_windows.append(normalized)

        # Batch predict
        if len(all_windows) > 0:
            x_batch = np.array(all_windows, dtype=np.float32)
            y_probs = self.model.predict(x_batch, verbose=0, batch_size=256)
            all_predictions = (y_probs[:, 0] > 0.5).astype(int).tolist()
        else:
            all_predictions = []

        # Post-processing: convert window predictions to sample-level weartime
        step = win_samples - overlap_samples

        (
            self.weartime_list_,
            self.total_weartime_samples_,
            _total_weartime_seconds,
            self.total_weartime_minutes_,
            self.total_weartime_hours_,
            self.total_weartime_hours_during_waking_,
            _coverage,
        ) = overlapping_windows_to_sample_labels(
            predictions=all_predictions,
            data_len=data_length,
            window_size=win_samples,
            stride=step,
            sampling_rate_hz=int(sampling_rate_hz),
            min_confidence_short_bouts=0.90,
            short_bout_threshold_minutes=20,
            min_bout_duration_seconds=15,
        )

        # Ensure end indices don't exceed data length
        self.weartime_list_["end"] = self.weartime_list_["end"].clip(upper=data_length)

        # Unify format (adds wt_id index, ensures correct dtypes)
        self.weartime_list_ = _unify_weartime_df(self.weartime_list_)

        return self
