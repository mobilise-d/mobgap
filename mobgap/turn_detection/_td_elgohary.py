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


def _merge_intervals(intervals: np.ndarray, max_distance: int) -> np.ndarray:
    # TODO: Remove when updating to new gaitmap version. Workaround for now.
    if len(intervals) == 0:
        return intervals
    return merge_intervals(intervals, max_distance)


def _as_valid_turn_list(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: Add type conversions
    df = df.reset_index(drop=True).rename_axis("turn_id")
    # To ensure that the index is 1-based
    df.index += 1
    return df

class TdElgohary(BaseTurnDetector):
    """
    performs turning detection according to El-Gohary et al.

    Parameters
    ----------
      fc_hz: cutoff freq for turning detection
      velocity_dps: threshold for turning velocity_dps [deg/s]
      height: minimal height for turning peaks
      concat_length: maximal distance of turning sequences to be concatenated [in s]
      allowed_turn_duration_s: minimal length of a turning sequence [in s]
      max_duration: maximal length of a turning sequence [in s]
      angle_threshold_degrees: tuple with threshold for detected turning angle

    Attributes
    ----------
      data_: gyr_z data
      complete_turns_list_: list of turning intervals with turning angles > angle_threshold_degrees,
                         each is represented as [start, length]
      suitable_angles_:  all corresponding turning angles of the turns in complete_turns_list
      all_turns_: all turns and their position within the signal
      all_angles_degrees: corresponding turning angles to all turns

    Notes
    -----
    COmpared to original publication:

        - We skip the global frame transformation and assume that the z axis roughly aligns with the turning axis.

    Compared to matlab implementation:

    """

    # parameters required for turning detection
    data_: np.ndarray
    fc_hz: float
    velocity_dps: float
    height: float
    concat_length: float
    allowed_turn_duration_s: tuple[float, float]
    angle_threshold_degrees: tuple[float, float]
    complete_turns_list_: list
    suitable_angles_: list
    all_turns_: list
    all_angles_degrees: list
    Turn_End_seconds: list
    Turn_Start_seconds: list
    duration_list_seconds: list
    duration_list_frames: list

    def __init__(
        self,
        smoothing_filter: BaseFilter = cf(ButterworthFilter(order=4, cutoff_freq_hz=0.5, filter_type="lowpass")),
        velocity_dps=5,
        height=15,
        concat_length=0.05,
        allowed_turn_duration_s: tuple[float, float] = (0.5, 10),
        angle_threshold_degrees: tuple[float, float] = (45, np.inf),
    ):
        self.smoothing_filter = smoothing_filter
        self.angle_threshold_degrees = angle_threshold_degrees
        self.allowed_turn_duration_s = allowed_turn_duration_s
        self.concat_length = concat_length
        self.height = height
        self.velocity_dps = velocity_dps


    def _return_empty(self):
        self.turn_list_ = pd.DataFrame(columns=["start", "end", "duration_s", "turn_angle_deg", "direction"])
        self.raw_turn_list_ = self.turn_list_.copy().assign(center=[])
        self.yaw_angle_ = pd.DataFrame(columns=["angle_deg"])
        return self

    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **kwargs: Unpack[dict[str, Any]]) -> Self:
        gyr_z = data["gyr_z"].to_numpy()
        filtered_gyr_z = self.smoothing_filter.clone().filter(gyr_z, sampling_rate_hz=sampling_rate_hz).filtered_data_
        abs_filtered_gyr_z = np.abs(filtered_gyr_z)
        dominant_peaks, _ = find_peaks(abs_filtered_gyr_z, height=self.height)

        if len(dominant_peaks) == 0:
            return self._return_empty()
        # Then we find the start and the end of the turn as the first and the last sample around each peak that is above
        # the velocity threshold
        # Note: The "abs" part here is missing in the original implementation
        above_threshold = abs_filtered_gyr_z < self.velocity_dps
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
            # For all left and all right turns, we merge turns that are closer than concat_length and have the same direction
            turns.groupby("direction")
            .apply(
                lambda df_: pd.DataFrame(merge_intervals(df_[["start", "end"]].to_numpy()), columns=["start", "end"])
            )
            # Then we combine all turns, sort them and recalculate the metrics.
            # Recalculation is required as the merging might have changed the start and end indices, and we can not simply
            # add the duration and angle columns, as we allow for merging of turns with a short break in between.
            # So we need to account for movement that happened in between.
            .sort_values(by="start")
            .reset_index(drop=True)
            .pipe(_calculate_metrics)
        )

        # Finally we filter based on the durations and the angles
        bool_map = turns["duration_s"].between(*self.allowed_turn_duration_s) & turns["turn_angle_deg"].abs().between(
            *self.angle_threshold_degrees
        )
        self.turn_list_ = _as_valid_turn_list(turns.loc[bool_map].copy())

        return self
