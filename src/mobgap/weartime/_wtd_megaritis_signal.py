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

import warnings
from typing import Any, Literal, Unpack

import numpy as np
import pandas as pd
from typing_extensions import Self

from mobgap._utils_internal.misc import timed_action_method
from mobgap.weartime.base import BaseWeartimeDetector, _unify_weartime_df, base_weartime_docfiller
from mobgap.weartime.utils.ml_feature_extraction import (
    extract_features_from_windows,
    remove_short_wear_bouts_by_ratio,
    rolling_window_indices,
)
from mobgap.weartime.utils.weartime_calc import generate_weartime_list_from_samples
from mobgap.weartime.utils.windows_to_weartime import remove_isolated_short_periods


@base_weartime_docfiller
class WtdMegaritisSignal(BaseWeartimeDetector):
    """
    Novel signal processing wear-time detection algorithm.

    The algorithm uses gyroscopes to detect angular (rotational) movement during true wear-time.
    Natural body movements show characteristic low-frequency rotational patterns (<15-17 Hz).
    Movement variability captures continuous micro-movements during wear, discriminating wear
    from non-wear independently of activity intensity.

    Parameters
    ----------
    window_min : int
        Macro window size in minutes (default: 60)
    step_min : int
        Macro window step in minutes (default: 15)
    window_size : int
        Micro window size in seconds (default: 5)
    overlap : float
        Micro window overlap fraction, 0.0-1.0 (default: 0.5)
    prob_thresh : float
        Probability threshold for macro-level non-wear decision (default: 0.4)
    gyr_ml_centroid_thresh_hz : float
        Threshold for ML gyroscope spectral centroid in Hz (default: 16.0)
    gyr_is_centroid_thresh_hz : float
        Threshold for IS gyroscope spectral centroid in Hz (default: 18.0)
    acc_pa_std_thresh : float
        Threshold for PA acceleration standard deviation (default: 0.17)
    voting_mode : bool
        If True, use voting system (default: True)
    min_features_required : int
        Minimum features meeting wear criteria (default: 2)
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
    diagnostics_ : dict
        Diagnostic information with 'macro' and 'sample_votes' keys

    Notes
    -----
    **Algorithm Workflow**
    1. Sliding macro windows are defined over the input data
    2. Each macro window is divided into micro windows (5s) for feature extraction
    3. Three features are extracted per micro window:
       gyr_ml_spectral_centroid (frequency of mediolateral rotation),
       gyr_is_spectral_centroid (frequency of vertical rotation),
       acc_pa_std (variability of anteroposterior acceleration)
    4. Each micro window is classified using 2-out-of-3 voting
    5. Macro-level decision via probability threshold (default 0.4)
    6. Per-sample votes accumulated from overlapping macro windows
    7. Final wear/non-wear determined by vote majority
    8. Two-stage post-processing removes artifacts

    Post-processing rationale: Stage 1 (15-second filter) removes brief isolated periods
    from sensor noise or voting conflicts. Stage 2 (20-minute ratio filter) removes short
    wear bouts surrounded by disproportionate non-wear (ratio <0.3), likely device handling
    rather than true wear events.

    **Waking Hours Calculation**

    In addition to total wear-time, this algorithm calculates wear-time during waking hours
    (07:00-22:00), required for Mobilise-D DMO weekly aggregation. The waking hours value is
    extracted from the post-processed sample-level predictions by filtering wear-time to the
    07:00-22:00 window.

    The pipeline is designed for daily recordings (midnight-to-midnight, ~24 hours).
    For recordings shorter than 22 hours or longer than 25 hours, the algorithm issues a warning
    and uses ``total_weartime_hours_`` as a fallback for ``total_weartime_hours_during_waking_``,
    as the waking hours window cannot be reliably identified in non-standard recording durations.
    Waking hours are identified using sample indices (07:00 = 7x3600xsampling_rate_hz) rather than
    timestamps, ensuring compatibility with devices that may not provide timestamp metadata.
    """

    # Type hints
    data_length: int
    diagnostics_: dict[str, pd.DataFrame | list]
    total_weartime_hours_during_waking_: float

    def __init__(
        self,
        *,
        window_min: int = 60,
        step_min: int = 15,
        window_size: int = 5,
        overlap: float = 0.5,
        prob_thresh: float = 0.4,
        gyr_ml_centroid_thresh_hz: float = 16.0,
        gyr_is_centroid_thresh_hz: float = 18.0,
        acc_pa_std_thresh: float = 0.17,
        voting_mode: bool = True,
        min_features_required: int = 2,
        position: Literal["lowback"] = "lowback",
    ) -> None:
        self.window_min = window_min
        self.step_min = step_min
        self.window_size = window_size
        self.overlap = overlap
        self.prob_thresh = prob_thresh
        self.gyr_ml_centroid_thresh_hz = gyr_ml_centroid_thresh_hz
        self.gyr_is_centroid_thresh_hz = gyr_is_centroid_thresh_hz
        self.acc_pa_std_thresh = acc_pa_std_thresh
        self.voting_mode = voting_mode
        self.min_features_required = min_features_required
        self.position = position

    @timed_action_method
    @base_weartime_docfiller
    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float = 100,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """
        %(detect_short)s using multi-level voting on gyroscope and accelerometer features.

        The algorithm processes data through three levels of decision-making plus two-stage post-processing:
        1. Micro-level: Feature-based classification (2/3 voting on gyro + accel features)
        2. Macro-level: Aggregation of micro windows via probability threshold
        3. Sample-level: Majority voting across overlapping macro windows
        4. Post-processing Stage 1: Remove brief isolated periods (<15 seconds)
        5. Post-processing Stage 2: Remove short wear bouts (≤20 min) with low contextual ratio (<0.3)

        Parameters
        ----------
        %(detect_para)s

        %(detect_return)s

        Additional Attributes
        ---------------------
        diagnostics_ : dict
            Diagnostic information containing:

            - 'macro': DataFrame with per-macro-window statistics
            - 'sample_votes': DataFrame with per-sample vote distributions
        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.data_length = len(data)
        self.diagnostics_: dict[str, pd.DataFrame | list] = {
            "macro": [],
            "sample_votes": pd.DataFrame(),
        }

        n_samples = len(self.data)

        # Macro window definition
        window_samples = int(self.window_min * 60 * self.sampling_rate_hz)
        step_samples = int(self.step_min * 60 * self.sampling_rate_hz)

        # --- per-sample vote counters ---
        wear_votes = np.zeros(n_samples, dtype=int)
        non_wear_votes = np.zeros(n_samples, dtype=int)

        # --- Handle very short recordings (shorter than one macro window) ---
        if n_samples < window_samples:
            # Process entire recording as a single partial macro window
            self._process_macro_window(
                data=data,
                start_idx=0,
                end_idx=n_samples,
                wear_votes=wear_votes,
                non_wear_votes=non_wear_votes,
                sampling_rate_hz=sampling_rate_hz,
                is_boundary=True,
                is_short_recording=True,
            )
        else:
            # --- Normal processing: Sliding macro windows ---
            for start_macro in range(0, n_samples - window_samples + 1, step_samples):
                end_macro = start_macro + window_samples

                self._process_macro_window(
                    data=data,
                    start_idx=start_macro,
                    end_idx=end_macro,
                    wear_votes=wear_votes,
                    non_wear_votes=non_wear_votes,
                    sampling_rate_hz=sampling_rate_hz,
                    is_boundary=False,
                )

            # --- Process boundary samples (partial macro window at end) ---
            # Calculate where the last complete macro window ended
            last_complete_macro_end = ((n_samples - window_samples) // step_samples + 1) * step_samples

            if last_complete_macro_end < n_samples:
                # There are unprocessed boundary samples
                # Create a partial macro window that ends at n_samples
                boundary_macro_start = max(0, n_samples - window_samples)

                self._process_macro_window(
                    data=data,
                    start_idx=boundary_macro_start,
                    end_idx=n_samples,
                    wear_votes=wear_votes,
                    non_wear_votes=non_wear_votes,
                    sampling_rate_hz=sampling_rate_hz,
                    is_boundary=True,
                )

        # --- FINAL decision per sample ---
        weartime_flags = (wear_votes >= non_wear_votes).astype(int)

        # Post-processing Stage 1: Remove very brief isolated periods (<15 seconds)
        # Removes sensor noise, voting edge effects, and transient artifacts
        weartime_flags = remove_isolated_short_periods(
            weartime_flags, min_period_sec=15.0, sampling_rate_hz=self.sampling_rate_hz
        )

        # Post-processing Stage 2: Remove short wear bouts (≤20 min) with low contextual ratio
        # Removes suspected device handling events (e.g., 10-min "wear" surrounded by hours of non-wear)
        weartime_flags = remove_short_wear_bouts_by_ratio(
            weartime_flags, max_bout_minutes=20.0, min_ratio=0.3, sampling_rate_hz=self.sampling_rate_hz
        )

        # Add diagnostic info after loop
        self.diagnostics_["macro"] = pd.DataFrame(self.diagnostics_["macro"])
        self.diagnostics_["sample_votes"] = pd.DataFrame(
            {
                "wear_votes": wear_votes,
                "non_wear_votes": non_wear_votes,
                "vote_margin": wear_votes - non_wear_votes,
                "final_flag": weartime_flags,
            }
        )

        # Output formatting (per sample converted to weartime list)
        self.weartime_list_ = generate_weartime_list_from_samples(weartime_flags)

        # Clip end to actual data length
        self.weartime_list_["end"] = self.weartime_list_["end"].clip(upper=self.data_length)

        # Unify format (adds wt_id index, ensures correct dtypes)
        self.weartime_list_ = _unify_weartime_df(self.weartime_list_)

        self.total_weartime_samples_ = (self.weartime_list_["end"] - self.weartime_list_["start"]).sum()
        self.total_weartime_minutes_ = self.total_weartime_samples_ / (60 * self.sampling_rate_hz)
        self.total_weartime_hours_ = self.total_weartime_samples_ / (3600 * self.sampling_rate_hz)

        # Weartime during waking hours (07:00-22:00)
        waking_start_sample = int(7 * 3600 * self.sampling_rate_hz)
        waking_end_sample = int(22 * 3600 * self.sampling_rate_hz)
        recording_hours = self.data_length / (3600 * self.sampling_rate_hz)

        # Check if recording is according to mobgap use case (single day)
        if self.data_length < waking_end_sample:
            # Recording shorter than 22:00 - use full weartime
            warnings.warn(
                f"Recording duration ({recording_hours:.1f}h) is shorter than waking hours window (07:00-22:00). "
                f"Using total_weartime_hours_ for weartime_during_waking_hours.",
                stacklevel=2,
            )
            self.total_weartime_hours_during_waking_ = self.total_weartime_hours_
        elif recording_hours > 25:
            # Recording longer than 25 hours - use full weartime
            warnings.warn(
                f"Recording duration ({recording_hours:.1f}h) exceeds a full day. "
                f"Waking hours calculation assumes the recording is segmented per day. "
                f"Using total_weartime_hours_ for weartime_during_waking_hours.",
                stacklevel=2,
            )
            self.total_weartime_hours_during_waking_ = self.total_weartime_hours_
        else:
            # Normal day (22-25h): crop to waking hours
            weartime_flags_waking = weartime_flags.copy()
            weartime_flags_waking[:waking_start_sample] = 0
            weartime_flags_waking[waking_end_sample:] = 0

            total_weartime_samples_waking = weartime_flags_waking.sum()
            self.total_weartime_hours_during_waking_ = total_weartime_samples_waking / (3600 * self.sampling_rate_hz)

        return self

    def _process_macro_window(
        self,
        data: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        wear_votes: np.ndarray,
        non_wear_votes: np.ndarray,
        sampling_rate_hz: float,
        is_boundary: bool = False,
        is_short_recording: bool = False,
    ) -> None:
        """
        Process a single macro window (complete or partial) and accumulate sample-level votes.

        Extracts micro windows, computes features, classifies each micro window, aggregates
        to macro-level decision, and assigns votes to all samples within the macro window.

        Parameters
        ----------
        data : pd.DataFrame
            Complete input data with accelerometer and gyroscope columns
        start_idx : int
            Start sample index of macro window
        end_idx : int
            End sample index of macro window (exclusive)
        wear_votes : np.ndarray
            Array to accumulate wear votes (modified in place)
        non_wear_votes : np.ndarray
            Array to accumulate non-wear votes (modified in place)
        sampling_rate_hz : float
            Sampling frequency in Hz
        is_boundary : bool, optional
            Whether this is a partial macro window at recording boundary (default: False)
        is_short_recording : bool, optional
            Whether entire recording is shorter than one macro window (default: False)
        """
        macro_window_data = data.iloc[start_idx:end_idx]
        n_samples_macro = len(macro_window_data)

        # Micro windows
        win_samples_micro = int(self.window_size * sampling_rate_hz)
        step = int(win_samples_micro * (1 - self.overlap))

        features_micro = []
        for start_micro, end_micro in rolling_window_indices(n_samples_macro, win_samples_micro, step):
            micro_window = macro_window_data.iloc[start_micro:end_micro]
            features_micro.append(extract_features_from_windows(micro_window, sampling_rate=sampling_rate_hz))

        # Only proceed if we have micro windows
        if len(features_micro) == 0:
            return

        features_micro_df = pd.DataFrame(features_micro)

        # --- Wear/Non-wear classification per micro window ---
        micro_wear_flags = self._classify_micro_windows(features_micro_df)

        # Macro-level decision: proportion of micro windows classified as non-wear
        micro_non_wear = ~micro_wear_flags
        macro_non_wear = micro_non_wear.mean() >= self.prob_thresh

        # --- per-sample vote assignment ---
        # All samples in this macro window receive one vote (wear or non-wear)
        if macro_non_wear:
            non_wear_votes[start_idx:end_idx] += 1
        else:
            wear_votes[start_idx:end_idx] += 1

        # Diagnostic info for this macro window
        macro_score = micro_non_wear.mean()
        self.diagnostics_["macro"].append(
            {
                "start": start_idx,
                "end": end_idx,
                "macro_score": macro_score,
                "macro_non_wear": macro_non_wear,
                "n_micro_windows": len(micro_non_wear),
                "micro_non_wear_rate": macro_score,
                "n_wear": micro_wear_flags.sum(),
                "n_non_wear": micro_non_wear.sum(),
                "is_boundary_window": is_boundary,
                "is_short_recording": is_short_recording,
            }
        )

    def _classify_micro_windows(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Classify micro windows as wear/non-wear based on feature thresholds.

        Evaluates three features per micro window:
        1. gyr_ml_spectral_centroid < threshold → wear
        2. gyr_is_spectral_centroid < threshold → wear
        3. acc_pa_std > threshold → wear

        Uses 2-out-of-3 voting (default) for robustness to individual sensor failures.

        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame with columns: gyr_ml_spectral_centroid, gyr_is_spectral_centroid, acc_pa_std.
            Features should be computed using extract_features_from_windows() with appropriate sampling_rate.

        Returns
        -------
        np.ndarray
            Boolean array where True = wear, False = non-wear.
            Windows with missing features (NaN) are conservatively classified as wear.
        """
        n_windows = len(features_df)
        wear_flags = np.zeros(n_windows, dtype=bool)

        for i in range(n_windows):
            # Check if required features are present (not NaN)
            if features_df.loc[i, ["gyr_ml_spectral_centroid", "gyr_is_spectral_centroid", "acc_pa_std"]].isna().any():
                # Default to wear if features are missing (conservative)
                wear_flags[i] = True
                continue

            # Extract feature values
            gyr_ml_centroid = features_df.loc[i, "gyr_ml_spectral_centroid"]
            gyr_is_centroid = features_df.loc[i, "gyr_is_spectral_centroid"]
            acc_pa_std = features_df.loc[i, "acc_pa_std"]

            # Sanity check: Handle theoretical all-zero feature case
            # When signal is all zeros, spectral centroids = 0 and acc_pa_std = 0.
            # Zero spectral centroids would incorrectly meet wear criteria (< threshold),
            # causing false positive detection. This scenario is impossible with real sensor data
            # (device noise always produces non-zero signal), but can occur in synthetic test data.
            # Algorithm was validated with this check on real-world data where this condition
            # never occurs.
            if gyr_ml_centroid == 0 and gyr_is_centroid == 0 and acc_pa_std == 0:
                wear_flags[i] = False  # Classifying as non-wear
                continue

            # Evaluate wear criteria for each feature
            gyr_ml_wear = gyr_ml_centroid < self.gyr_ml_centroid_thresh_hz
            gyr_is_wear = gyr_is_centroid < self.gyr_is_centroid_thresh_hz
            acc_pa_wear = acc_pa_std > self.acc_pa_std_thresh

            if self.voting_mode:
                # Voting system: count how many features meet wear criteria
                wear_score = int(gyr_ml_wear) + int(gyr_is_wear) + int(acc_pa_wear)
                wear_flags[i] = wear_score >= self.min_features_required
            else:
                # Strict AND: all features must meet wear criteria
                wear_flags[i] = gyr_ml_wear and gyr_is_wear and acc_pa_wear

        return wear_flags
