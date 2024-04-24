from typing import Any

import numpy as np
import pandas as pd
from gaitmap.utils.array_handling import merge_intervals
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks
from tpcp import cf
from typing_extensions import Self, Unpack

from mobgap.data_transform import ButterworthFilter
from mobgap.data_transform.base import BaseFilter
from mobgap.turn_detection.base import BaseTurnDetector


def _as_valid_turn_list(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: Add type conversions
    df = df.reset_index(drop=True).rename_axis("turn_id")
    # To ensure that the index is 1-based
    df.index += 1
    return df


class TdElGohary(BaseTurnDetector):
    """Detect turns in the continous yaw signal of a lower back IMU based on the algorithm by El-Gohary et al.

    This algorithm uses the lowpass filtered yaw-signal, then identifies peaks in its norm higher than
    ``min_peak_angle_velocity_dps``.
    Each peak is considered a turn-candidate and the start and end of each turn is identified as the first and last
    samples before and after the respective peak that are below a certain threshold (``lower_threshold_velocity_dps``).
    Consecutive turns candidates in the same direction that are closer than ``min_gap_between_turns_s`` are merged.
    The final turns are then filtered based on their duration and the turning angle.

    Parameters
    ----------
    smoothing_filter
        The filter to apply to the yaw signal before detecting the turns.
        That should be a lowpass filter, only preserving the main movements corresponding to the walking pattern.
    min_peak_angle_velocity_dps
        The minimum angular velocity in degrees per second that a peak in the yaw signal must have to be considered a
        turn candidate.
    lower_threshold_velocity_dps
        The threshold in degrees per second below which the signal is considered to be not turning.
        The algorithm finds the start and end of a turn as the first and last sample around each peak that is below this
        threshold.
    min_gap_between_turns_s
        The minimum gap in seconds between two turns to be considered separate turns.
        If two turn candidates that turn in the same direction are closer than this, they are merged.
    allowed_turn_duration_s
        The allowed duration of a turn in seconds.
        This is evaluated as the final step of the algorithm.
    allowed_turn_angle_deg
        The allowed turning angle of a turn in degrees.
        This is evaluated as the final step of the algorithm.


    Attributes
    ----------
    turn_list_

    Notes
    -----
    Implementation Details:

    - If a turn has no crossing with ``lower_threshold_velocity_dps``, as it happens close to the start or the end of
      the signal, the start/ends of the turn are replaced with the start/end of the signal.
      This might not always result in accurate turn angles for these turns, and for certain cases you might want to
      remove these turns from the list.
    - The turn angle is calculated by integrating the raw unfiltered signal.

    COmpared to original publication:

        - We skip the global frame transformation and assume that the z axis roughly aligns with the turning axis.

    Compared to matlab implementation:
        - We fix a bug, that only the start and the end of a turn was detected on the normal signal instead of the
          absolute signal.
          This made it "harder" to detect turns in one direction than the other.
    """

    smoothing_filter: BaseFilter

    raw_turn_list_: pd.DataFrame
    yaw_angle_: pd.DataFrame

    def __init__(
        self,
        *,
        smoothing_filter: BaseFilter = cf(ButterworthFilter(order=4, cutoff_freq_hz=0.5, filter_type="lowpass")),
        min_peak_angle_velocity_dps: float = 15,
        lower_threshold_velocity_dps: float = 5,
        min_gap_between_turns_s: float = 0.05,
        allowed_turn_duration_s: tuple[float, float] = (0.5, 10),
        allowed_turn_angle_deg: tuple[float, float] = (45, np.inf),
    ):
        self.smoothing_filter = smoothing_filter
        self.allowed_turn_angle_deg = allowed_turn_angle_deg
        self.allowed_turn_duration_s = allowed_turn_duration_s
        self.min_gap_between_turns_s = min_gap_between_turns_s
        self.min_peak_angle_velocity_dps = min_peak_angle_velocity_dps
        self.lower_threshold_velocity_dps = lower_threshold_velocity_dps

    def _return_empty(self):
        self.turn_list_ = pd.DataFrame(columns=["start", "end", "duration_s", "turn_angle_deg", "direction"])
        self.raw_turn_list_ = self.turn_list_.copy().assign(center=[])
        self.yaw_angle_ = pd.DataFrame(columns=["angle_deg"])
        return self

    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **kwargs: Unpack[dict[str, Any]]) -> Self:
        gyr_z = data["gyr_z"].to_numpy()
        filtered_gyr_z = self.smoothing_filter.clone().filter(gyr_z, sampling_rate_hz=sampling_rate_hz).filtered_data_
        # Note: The "abs" part here is missing in the original implementation.
        abs_filtered_gyr_z = np.abs(filtered_gyr_z)
        dominant_peaks, _ = find_peaks(abs_filtered_gyr_z, height=self.min_peak_angle_velocity_dps)

        if len(dominant_peaks) == 0:
            return self._return_empty()
        # Then we find the start and the end of the turn as the first and the last sample around each peak that is above
        # the velocity threshold
        above_threshold = abs_filtered_gyr_z < self.lower_threshold_velocity_dps
        crossings = np.where(np.diff(above_threshold.astype(int)) != 0)[0]

        if len(crossings) == 0:
            return self._return_empty()

        # Vectorized search for start indices
        start_indices = np.searchsorted(crossings, dominant_peaks, side="left") - 1
        # When the index is negative (i.e. no value was before it, we set the start index to 0, aka the beginning of
        # the WB)
        starts = np.where(start_indices < 0, 0, crossings[start_indices.clip(0, len(crossings) - 1)])

        # Vectorized search for end indices
        end_indices = np.searchsorted(crossings, dominant_peaks, side="right")
        # When the index is out of bounds (i.e. no value was after it, we set the end index to the end of the WB)
        ends = np.where(
            end_indices >= len(crossings), len(data) - 1, crossings[end_indices.clip(0, len(crossings) - 1)]
        )
        # To make the end index inclusive, we add 1
        ends += 1

        yaw_angle = cumulative_trapezoid(gyr_z, dx=1 / sampling_rate_hz, initial=0)
        self.yaw_angle_ = pd.DataFrame({"angle_deg": yaw_angle}, index=data.index)

        def _calculate_metrics(df):
            return df.assign(
                duration_s=lambda df_: (df_["end"] - df_["start"]) / sampling_rate_hz,
                turn_angle_deg=lambda df_: yaw_angle[df_["end"] - 1] - yaw_angle[df_["start"]],
                direction=lambda df_: np.sign(df_["turn_angle_deg"]),
                # TODO: Double check the sign directions
            ).replace({"direction": {-1: "left", 1: "right"}})

        # Create an array of turns
        turns = pd.DataFrame({"start": starts, "end": ends}).pipe(_calculate_metrics)
        self.raw_turn_list_ = _as_valid_turn_list(turns.copy().assign(center=dominant_peaks.astype(int)))

        turns = (
            # For all left and all right turns, we merge turns that are closer than min_gap_between_turns_s and have the
            # same direction
            turns.groupby("direction")
            .apply(
                lambda df_: pd.DataFrame(merge_intervals(df_[["start", "end"]].to_numpy()), columns=["start", "end"])
            )
            # Then we combine all turns, sort them and recalculate the metrics.
            # Recalculation is required as the merging might have changed the start and end indices, and we can not
            # simply add the duration and angle columns, as we allow for merging of turns with a short break in between.
            # So we need to account for movement that happened in between.
            .sort_values(by="start")
            .reset_index(drop=True)
            .pipe(_calculate_metrics)
        )

        # Finally we filter based on the durations and the angles
        bool_map = turns["duration_s"].between(*self.allowed_turn_duration_s) & turns["turn_angle_deg"].abs().between(
            *self.allowed_turn_angle_deg
        )
        self.turn_list_ = _as_valid_turn_list(turns.loc[bool_map].copy())

        return self
