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
from numbers import Integral
from typing import Any, Union

import numpy as np
import pandas as pd
from scipy.signal import welch
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import timed_action_method
from mobgap.weartime.base import BaseWeartimeDetector, _unify_weartime_df, base_weartime_docfiller
from mobgap.weartime.utils.ml_feature_extraction import remove_short_wear_bouts_by_ratio
from mobgap.weartime.utils.weartime_calc import generate_weartime_list_from_samples
from mobgap.weartime.utils.windows_to_weartime import remove_isolated_short_periods


def _window_starts(n_samples: int, window_samples: int, step_samples: int) -> np.ndarray:
    if n_samples < window_samples:
        return np.array([], dtype=np.int64)
    return np.arange(0, n_samples - window_samples + 1, step_samples, dtype=np.int64)


def _spectral_centroid_batched(
    signal: np.ndarray,
    starts: np.ndarray,
    offsets: np.ndarray,
    *,
    sampling_rate_hz: float,
) -> np.ndarray:
    windows = signal[starts[:, None] + offsets]
    frequencies, power = welch(windows, fs=sampling_rate_hz, nperseg=len(offsets), axis=1)
    total_power = power.sum(axis=1)
    weighted_power = (power * frequencies).sum(axis=1)
    centroid = np.zeros(len(starts), dtype=np.float64)
    np.divide(weighted_power, total_power, out=centroid, where=total_power > 0)
    return centroid


def _validate_waking_hours_min(waking_hours_min: tuple[int, int]) -> tuple[int, int]:
    try:
        start_min, end_min = waking_hours_min
    except (TypeError, ValueError) as exc:
        raise ValueError("`waking_hours_min` must be a tuple with exactly two values: `(start_min, end_min)`.") from exc

    if not isinstance(start_min, Integral) or not isinstance(end_min, Integral):
        raise TypeError("`waking_hours_min` values must be integer minutes since midnight.")
    if not 0 <= start_min < end_min <= 24 * 60:
        raise ValueError(
            "`waking_hours_min` must define a non-empty window within one day using minutes since midnight."
        )
    return int(start_min), int(end_min)


def _format_time_window(start_min: int, end_min: int) -> str:
    start_h, start_m = divmod(start_min, 60)
    end_h, end_m = divmod(end_min, 60)
    return f"{start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d}"


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
    waking_hours_min : tuple[int, int]
        Waking-hours window used for ``total_weartime_hours_during_waking_`` as ``(start, end)`` in minutes since
        midnight.
    feature_batch_size : int
        Number of 5-second windows processed together during feature extraction. Larger batches reduce overhead, while
        smaller batches reduce peak memory use.

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
    2. The complete macro windows share a global 5-second micro-window grid
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
    (07:00-22:00 by default), required for Mobilise-D DMO weekly aggregation. The waking hours value is
    extracted from the post-processed sample-level predictions by filtering wear-time to the
    configured waking-hours window.

    The pipeline is designed for daily recordings (midnight-to-midnight, ~24 hours).
    For recordings shorter than 22 hours or longer than 25 hours, the algorithm issues a warning
    and uses ``total_weartime_hours_`` as a fallback for ``total_weartime_hours_during_waking_``,
    as the waking hours window cannot be reliably identified in non-standard recording durations.
    Waking hours are identified using sample indices derived from minutes since midnight rather than timestamps,
    ensuring compatibility with devices that may not provide timestamp metadata.
    """

    diagnostics_: dict[str, Union[pd.DataFrame, list]]
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
        waking_hours_min: tuple[int, int] = (7 * 60, 22 * 60),
        feature_batch_size: int = 4096,
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
        self.waking_hours_min = waking_hours_min
        self.feature_batch_size = feature_batch_size

    @timed_action_method
    @base_weartime_docfiller
    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
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
        data_length = len(data)
        self.diagnostics_ = {"macro": [], "sample_votes": pd.DataFrame()}
        waking_start_min, waking_end_min = _validate_waking_hours_min(self.waking_hours_min)

        window_samples = int(self.window_min * 60 * self.sampling_rate_hz)
        step_samples = int(self.step_min * 60 * self.sampling_rate_hz)
        micro_window_samples = int(self.window_size * self.sampling_rate_hz)
        micro_step_samples = int(micro_window_samples * (1 - self.overlap))

        if micro_step_samples <= 0:
            raise ValueError("The micro-window step must be positive. Check `window_size` and `overlap`.")
        if self.feature_batch_size <= 0:
            raise ValueError("`feature_batch_size` must be a positive integer.")

        # Difference arrays let each macro-window decision vote for a full sample range without materializing all
        # overlapping per-sample updates immediately.
        wear_vote_diff = np.zeros(data_length + 1, dtype=np.int32)
        non_wear_vote_diff = np.zeros(data_length + 1, dtype=np.int32)

        if data_length < window_samples:
            self._process_single_macro_window(
                data=data,
                start_idx=0,
                end_idx=data_length,
                wear_vote_diff=wear_vote_diff,
                non_wear_vote_diff=non_wear_vote_diff,
                sampling_rate_hz=sampling_rate_hz,
                micro_window_samples=micro_window_samples,
                micro_step_samples=micro_step_samples,
                is_boundary=True,
                is_short_recording=True,
            )
        else:
            complete_macro_starts = _window_starts(data_length, window_samples, step_samples)
            self._process_complete_macro_windows(
                data=data,
                macro_starts=complete_macro_starts,
                window_samples=window_samples,
                micro_window_samples=micro_window_samples,
                micro_step_samples=micro_step_samples,
                wear_vote_diff=wear_vote_diff,
                non_wear_vote_diff=non_wear_vote_diff,
                sampling_rate_hz=sampling_rate_hz,
            )

            last_complete_macro_end = ((data_length - window_samples) // step_samples + 1) * step_samples
            if last_complete_macro_end < data_length:
                # The final boundary window is anchored to the recording end and can be shifted relative to the shared
                # global micro-window grid, so it is classified separately.
                self._process_single_macro_window(
                    data=data,
                    start_idx=max(0, data_length - window_samples),
                    end_idx=data_length,
                    wear_vote_diff=wear_vote_diff,
                    non_wear_vote_diff=non_wear_vote_diff,
                    sampling_rate_hz=sampling_rate_hz,
                    micro_window_samples=micro_window_samples,
                    micro_step_samples=micro_step_samples,
                    is_boundary=True,
                    is_short_recording=False,
                )

        wear_votes = np.cumsum(wear_vote_diff[:-1])
        non_wear_votes = np.cumsum(non_wear_vote_diff[:-1])
        # Keep the original conservative tie-breaking behavior: equal votes are treated as wear.
        weartime_flags = (wear_votes >= non_wear_votes).astype(int)

        # Stage 1 removes brief isolated periods caused by sensor noise, voting edge effects, or transient artifacts.
        weartime_flags = remove_isolated_short_periods(
            weartime_flags, min_period_sec=15.0, sampling_rate_hz=self.sampling_rate_hz
        )
        # Stage 2 removes short wear bouts that are likely device handling rather than sustained wear.
        weartime_flags = remove_short_wear_bouts_by_ratio(
            weartime_flags, max_bout_minutes=20.0, min_ratio=0.3, sampling_rate_hz=self.sampling_rate_hz
        )

        self.diagnostics_["macro"] = pd.DataFrame(self.diagnostics_["macro"])
        self.diagnostics_["sample_votes"] = pd.DataFrame(
            {
                "wear_votes": wear_votes,
                "non_wear_votes": non_wear_votes,
                "vote_margin": wear_votes - non_wear_votes,
                "final_flag": weartime_flags,
            }
        )

        self.weartime_list_ = generate_weartime_list_from_samples(weartime_flags)
        self.weartime_list_["end"] = self.weartime_list_["end"].clip(upper=data_length)
        self.weartime_list_ = _unify_weartime_df(self.weartime_list_)

        self.total_weartime_samples_ = (self.weartime_list_["end"] - self.weartime_list_["start"]).sum()
        self.total_weartime_minutes_ = self.total_weartime_samples_ / (60 * self.sampling_rate_hz)
        self.total_weartime_hours_ = self.total_weartime_samples_ / (3600 * self.sampling_rate_hz)
        self._set_waking_hours_summary(weartime_flags, data_length, waking_start_min, waking_end_min)

        return self

    def _process_complete_macro_windows(
        self,
        *,
        data: pd.DataFrame,
        macro_starts: np.ndarray,
        window_samples: int,
        micro_window_samples: int,
        micro_step_samples: int,
        wear_vote_diff: np.ndarray,
        non_wear_vote_diff: np.ndarray,
        sampling_rate_hz: float,
    ) -> None:
        if len(macro_starts) == 0:
            return

        n_micro_per_macro = len(_window_starts(window_samples, micro_window_samples, micro_step_samples))
        if n_micro_per_macro == 0:
            return

        first_micro_start = int(macro_starts[0])
        last_micro_start = int(macro_starts[-1] + (n_micro_per_macro - 1) * micro_step_samples)
        # Complete macro windows all start on the same step grid, so their overlapping micro windows can be classified
        # once globally and reused for each macro decision.
        global_micro_starts = np.arange(
            first_micro_start,
            last_micro_start + micro_step_samples,
            micro_step_samples,
            dtype=np.int64,
        )
        micro_wear_flags = self._classify_micro_windows_from_starts(
            data=data,
            starts=global_micro_starts,
            window_samples=micro_window_samples,
            sampling_rate_hz=sampling_rate_hz,
        )
        micro_non_wear = ~micro_wear_flags
        non_wear_prefix = np.concatenate([[0], np.cumsum(micro_non_wear, dtype=np.int64)])

        for start_idx in macro_starts:
            end_idx = int(start_idx + window_samples)
            micro_start_idx = int((start_idx - first_micro_start) // micro_step_samples)
            n_non_wear = int(non_wear_prefix[micro_start_idx + n_micro_per_macro] - non_wear_prefix[micro_start_idx])
            self._add_macro_decision(
                start_idx=int(start_idx),
                end_idx=end_idx,
                n_micro_windows=n_micro_per_macro,
                n_non_wear=n_non_wear,
                wear_vote_diff=wear_vote_diff,
                non_wear_vote_diff=non_wear_vote_diff,
                is_boundary=False,
                is_short_recording=False,
            )

    def _process_single_macro_window(
        self,
        *,
        data: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        wear_vote_diff: np.ndarray,
        non_wear_vote_diff: np.ndarray,
        sampling_rate_hz: float,
        micro_window_samples: int,
        micro_step_samples: int,
        is_boundary: bool,
        is_short_recording: bool,
    ) -> None:
        relative_micro_starts = _window_starts(end_idx - start_idx, micro_window_samples, micro_step_samples)
        if len(relative_micro_starts) == 0:
            return

        micro_wear_flags = self._classify_micro_windows_from_starts(
            data=data,
            starts=relative_micro_starts + start_idx,
            window_samples=micro_window_samples,
            sampling_rate_hz=sampling_rate_hz,
        )
        self._add_macro_decision(
            start_idx=start_idx,
            end_idx=end_idx,
            n_micro_windows=len(micro_wear_flags),
            n_non_wear=int((~micro_wear_flags).sum()),
            wear_vote_diff=wear_vote_diff,
            non_wear_vote_diff=non_wear_vote_diff,
            is_boundary=is_boundary,
            is_short_recording=is_short_recording,
        )

    def _classify_micro_windows_from_starts(
        self,
        *,
        data: pd.DataFrame,
        starts: np.ndarray,
        window_samples: int,
        sampling_rate_hz: float,
    ) -> np.ndarray:
        wear_flags = np.empty(len(starts), dtype=bool)
        if len(starts) == 0:
            return wear_flags

        required_columns = {"acc_pa", "gyr_ml", "gyr_is"}
        if not required_columns.issubset(data.columns):
            # Missing channels are treated as inconclusive and therefore as wear, matching the original conservative
            # handling of missing feature values.
            wear_flags[:] = True
            return wear_flags

        acc_pa = data["acc_pa"].to_numpy(copy=False)
        gyr_ml = data["gyr_ml"].to_numpy(copy=False)
        gyr_is = data["gyr_is"].to_numpy(copy=False)
        offsets = np.arange(window_samples, dtype=np.int64)

        for batch_start in range(0, len(starts), self.feature_batch_size):
            batch = starts[batch_start : batch_start + self.feature_batch_size]
            batch_offsets = batch[:, None] + offsets
            acc_pa_std = np.std(acc_pa[batch_offsets], axis=1, ddof=1)
            gyr_ml_centroid = _spectral_centroid_batched(gyr_ml, batch, offsets, sampling_rate_hz=sampling_rate_hz)
            gyr_is_centroid = _spectral_centroid_batched(gyr_is, batch, offsets, sampling_rate_hz=sampling_rate_hz)
            wear_flags[batch_start : batch_start + len(batch)] = self._classify_feature_arrays(
                acc_pa_std=acc_pa_std,
                gyr_ml_centroid=gyr_ml_centroid,
                gyr_is_centroid=gyr_is_centroid,
            )

        return wear_flags

    def _classify_feature_arrays(
        self,
        *,
        acc_pa_std: np.ndarray,
        gyr_ml_centroid: np.ndarray,
        gyr_is_centroid: np.ndarray,
    ) -> np.ndarray:
        missing_features = np.isnan(acc_pa_std) | np.isnan(gyr_ml_centroid) | np.isnan(gyr_is_centroid)
        all_zero_features = (gyr_ml_centroid == 0) & (gyr_is_centroid == 0) & (acc_pa_std == 0)

        gyr_ml_wear = gyr_ml_centroid < self.gyr_ml_centroid_thresh_hz
        gyr_is_wear = gyr_is_centroid < self.gyr_is_centroid_thresh_hz
        acc_pa_wear = acc_pa_std > self.acc_pa_std_thresh

        if self.voting_mode:
            wear_score = gyr_ml_wear.astype(np.int8) + gyr_is_wear.astype(np.int8) + acc_pa_wear.astype(np.int8)
            wear_flags = wear_score >= self.min_features_required
        else:
            wear_flags = gyr_ml_wear & gyr_is_wear & acc_pa_wear

        wear_flags[missing_features] = True
        # All-zero synthetic signals have zero spectral centroids and zero acceleration variance. The centroid checks
        # alone would otherwise classify them as wear, although a constant signal should be non-wear.
        wear_flags[all_zero_features & ~missing_features] = False
        return wear_flags

    def _add_macro_decision(
        self,
        *,
        start_idx: int,
        end_idx: int,
        n_micro_windows: int,
        n_non_wear: int,
        wear_vote_diff: np.ndarray,
        non_wear_vote_diff: np.ndarray,
        is_boundary: bool,
        is_short_recording: bool,
    ) -> None:
        n_wear = n_micro_windows - n_non_wear
        macro_score = n_non_wear / n_micro_windows
        macro_non_wear = macro_score >= self.prob_thresh

        if macro_non_wear:
            non_wear_vote_diff[start_idx] += 1
            non_wear_vote_diff[end_idx] -= 1
        else:
            wear_vote_diff[start_idx] += 1
            wear_vote_diff[end_idx] -= 1

        self.diagnostics_["macro"].append(
            {
                "start": start_idx,
                "end": end_idx,
                "macro_score": macro_score,
                "macro_non_wear": macro_non_wear,
                "n_micro_windows": n_micro_windows,
                "micro_non_wear_rate": macro_score,
                "n_wear": n_wear,
                "n_non_wear": n_non_wear,
                "is_boundary_window": is_boundary,
                "is_short_recording": is_short_recording,
            }
        )

    def _set_waking_hours_summary(
        self, weartime_flags: np.ndarray, data_length: int, waking_start_min: int, waking_end_min: int
    ) -> None:
        waking_start_sample = int(waking_start_min * 60 * self.sampling_rate_hz)
        waking_end_sample = int(waking_end_min * 60 * self.sampling_rate_hz)
        recording_hours = data_length / (3600 * self.sampling_rate_hz)
        waking_window = _format_time_window(waking_start_min, waking_end_min)

        if data_length < waking_end_sample:
            # The sample-index based waking-hours calculation assumes a complete enough daily recording.
            warnings.warn(
                f"Recording duration ({recording_hours:.1f}h) is shorter than waking hours window ({waking_window}). "
                f"Using total_weartime_hours_ for weartime_during_waking_hours.",
                stacklevel=2,
            )
            self.total_weartime_hours_during_waking_ = self.total_weartime_hours_
        elif recording_hours > 25:
            # Multi-day recordings need to be segmented before applying a single daily waking-hours window.
            warnings.warn(
                f"Recording duration ({recording_hours:.1f}h) exceeds a full day. "
                f"Waking hours calculation assumes the recording is segmented per day. "
                f"Using total_weartime_hours_ for weartime_during_waking_hours.",
                stacklevel=2,
            )
            self.total_weartime_hours_during_waking_ = self.total_weartime_hours_
        else:
            weartime_flags_waking = weartime_flags.copy()
            weartime_flags_waking[:waking_start_sample] = 0
            weartime_flags_waking[waking_end_sample:] = 0
            total_weartime_samples_waking = weartime_flags_waking.sum()
            self.total_weartime_hours_during_waking_ = total_weartime_samples_waking / (3600 * self.sampling_rate_hz)
