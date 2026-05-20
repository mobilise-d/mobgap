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

import pickle
from importlib.resources import files
from typing import Any, Literal, Unpack

import pandas as pd
from typing_extensions import Self

from mobgap._utils_internal.misc import timed_action_method
from mobgap.weartime.base import BaseWeartimeDetector, _unify_weartime_df, base_weartime_docfiller
from mobgap.weartime.utils.feature_extraction import extract_features_90pct, extract_full_features
from mobgap.weartime.utils.ml_feature_extraction import rolling_window_indices
from mobgap.weartime.utils.windows_to_weartime import overlapping_windows_to_sample_labels


@base_weartime_docfiller
class WtdMegaritisXGBoost(BaseWeartimeDetector):
    """
    XGBoost-based weartime detection for lower-back worn IMU sensors.

    Uses pre-trained XGBoost models with time-domain and frequency-domain features
    extracted from overlapping 5-second windows. Includes biomechanically-informed
    post-processing to filter short bouts and apply confidence thresholds.

    Two model variants are available:
    - Full: 230 features
    - Lightweight: 79 features (90%% SHAP importance), faster inference

    Post-processing steps:
    1. Majority voting across overlapping windows to obtain sample-level predictions
    2. Removal of wear bouts shorter than 15 seconds (biomechanically implausible)
    3. Confidence filtering for wear bouts under 20 minutes (requires >90%% vote agreement)
    4. Merging of short non-wear gaps (<15s) between wear periods

    Parameters
    ----------
    window_sec : float
        Window size in seconds for feature extraction (default: 5.0)
    overlap : float
        Window overlap fraction, 0.0 to 1.0 (default: 0.75)
    version : Literal["full", "lightweight"]
        Model variant: "full" (230 features) or "lightweight" (79 features, default)
    position : Literal['lowback']
        Sensor position (default: 'lowback', only supported position)

    Other Parameters
    ----------------
    %(other_parameters)s
    model : xgboost.XGBClassifier
        Pre-trained XGBoost model loaded during initialization
    feature_names : list
        List of feature names used by the model

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
    **Model and Performance**
    Pre-trained models are loaded from the package's production_models folder.
    XGBoost models do not require feature scaling.
    Feature extraction dominates computation time. For large datasets,
    consider using version="lightweight" for ~3x faster inference.

    **Waking Hours Calculation**

    In addition to total wear-time, this algorithm calculates wear-time during waking hours
    (07:00-22:00), required for Mobilise-D DMO weekly aggregation. The waking hours value is
    extracted from the post-processed sample-level predictions by filtering wear-time to the
    07:00-22:00 window.

    The pipeline is designed for daily recordings (midnight-to-midnight, ~24 hours).
    For recordings shorter than 22 hours or longer than 25 hours, the algorithm issues a warning
    and uses ``total_weartime_hours_`` as a fallback for ``total_weartime_hours_during_waking_``,
    as the waking hours window cannot be reliably identified in non-standard recording durations.
    Waking hours are identified using sample indices (07:00 = 7×3600×sampling_rate_hz) rather than
    timestamps, ensuring compatibility with devices that may not provide timestamp metadata.
    """

    # Type hints
    data_length: int
    feature_names: list[str]
    total_weartime_hours_during_waking_: float

    def __init__(
        self,
        *,
        window_sec: float = 5.0,
        overlap: float = 0.75,
        version: Literal["full", "lightweight"] = "lightweight",
        position: Literal["lowback"] = "lowback",
    ) -> None:
        self.window_sec = window_sec
        self.overlap = overlap
        self.version = version
        self.position = position

        # Load models once during initialization
        if self.version == "full":
            model_file = files("mobgap.weartime.production_models").joinpath("xgboost_fullfeatures_lowback_model.pkl")
            feature_order_file = files("mobgap.weartime.production_models").joinpath(
                "xgboost_fullfeatures_lowback_feature_order.pkl"
            )
        else:  # lightweight
            model_file = files("mobgap.weartime.production_models").joinpath("xgboost_90pct_lowback_model.pkl")
            feature_order_file = files("mobgap.weartime.production_models").joinpath(
                "xgboost_90pct_lowback_feature_order.pkl"
            )

        with model_file.open("rb") as f:
            self.model = pickle.load(f)

        with feature_order_file.open("rb") as f:
            self.feature_names = pickle.load(f)

    @timed_action_method
    @base_weartime_docfiller
    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float = 100, **_: Unpack[dict[str, Any]]) -> Self:
        """
        %(detect_short)s using XGBoost classifier with overlapping windows.

        Processes IMU data in overlapping windows, extracts features, applies the
        pre-trained XGBoost model, and converts window-level predictions to
        sample-level wear-time segments using majority voting and biomechanical
        post-processing rules.

        Parameters
        ----------
        %(detect_para)s

        %(detect_return)s

        Notes
        -----
        Post-processing pipeline:

        1. Majority voting: Each sample receives votes from overlapping windows
        2. Short bout removal: Wear bouts <15s are removed (too short to don/doff)
        3. Confidence filter: Wear bouts <20min require ≥90%% vote agreement
           (boundary bouts at start/end of data are exempt)
        4. Gap merging: Non-wear gaps <15s between wear periods are merged
        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.data_length = len(data)

        win_samples = int(self.window_sec * self.sampling_rate_hz)
        step = int(win_samples * (1 - self.overlap))
        n_samples = self.data_length

        # Store predictions for all windows
        all_predictions = []
        all_probabilities = []

        # Loop over windows
        for start, end in rolling_window_indices(n_samples, win_samples, step):
            win = self.data.iloc[start:end]

            # Extract features
            features_dict = extract_full_features(win) if self.version == "full" else extract_features_90pct(win)

            # Predict
            features_df = pd.DataFrame([features_dict])

            # Reorder columns to match training order
            features_df = features_df[self.feature_names]

            y_pred = self.model.predict(features_df)[0]
            y_prob = self.model.predict_proba(features_df)[:, 1][0]

            all_predictions.append(y_pred)
            all_probabilities.append(y_prob)

        # Post-processing: convert window predictions to sample-level weartime
        (
            self.weartime_list_,
            self.total_weartime_samples_,
            _total_seconds,
            self.total_weartime_minutes_,
            self.total_weartime_hours_,
            self.total_weartime_hours_during_waking_,
            _coverage,
        ) = overlapping_windows_to_sample_labels(
            predictions=all_predictions,
            data_len=self.data_length,
            window_size=win_samples,
            stride=step,
            sampling_rate_hz=int(sampling_rate_hz),
            )

        # Clip end to actual data length
        self.weartime_list_["end"] = self.weartime_list_["end"].clip(upper=self.data_length)

        # Unify format (adds wt_id index, ensures correct dtypes)
        self.weartime_list_ = _unify_weartime_df(self.weartime_list_)

        return self
