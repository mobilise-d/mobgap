from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks
from tpcp import cf
from typing_extensions import Self, Unpack

from mobgap.data_transform import ButterworthFilter
from mobgap.data_transform.base import BaseFilter
from mobgap.orientation_estimation.base import BaseOrientationEstimation
from mobgap.turning.base import BaseTurnDetector, base_turning_docfiller
from mobgap.utils.array_handling import merge_intervals
from mobgap.utils.conversions import as_samples
from mobgap.utils.dtypes import get_frame_definition

_turn_df_types = {
    "turn_id": "int64",
    "start": "int64",
    "end": "int64",
    "center": "int64",
    "duration_s": "float64",
    "angle_deg": "float64",
    "direction": pd.CategoricalDtype(categories=["left", "right"]),
}


def _as_valid_turn_list(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.reset_index(drop=True).rename_axis("turn_id").reset_index()
    dtypes_for_check = {col: _turn_df_types[col] for col in tmp.columns}
    return tmp.astype(dtypes_for_check)[list(dtypes_for_check.keys())].set_index("turn_id")


@base_turning_docfiller
class TdElGohary(BaseTurnDetector):
    """Detect turns in the continuous yaw signal of a lower back IMU based on the algorithm by El-Gohary et al.

    .. warning:: There are some issues with matching the results of the ElGohary algorithm to the reference system.
             The performance, we are observing here is far below the expected performance.
             We are investigating this currently, but until then, we recommend to do your own testing and validation
             before using this algorithm in production.

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
    orientation_estimation
        An optional instance of an orientation estimation algorithm to transform the IMU signal to the global frame.
        The version used in Mobilise-D did not use this, but the version described in the original publication did apply
        the algorithm to data from the global frame.
        This also makes the ``global_frame_data_`` attribute available, which contains the transformed data.


    Attributes
    ----------
    %(turn_list_)s
    raw_turn_list_
        The detected turns before filtering based on duration and angle.
        This df also contains an additional ``center`` column, which marks the position of the originally detected peak.
        This might be helpful for debugging or further analysis.
    yaw_angle_
        The yaw angle of the IMU signal estimated through the integration of the ``gyr_is`` signal.
        This is used to estimate the turn angle.
        This might be helpful for debugging or further analysis.
    global_frame_data_
        The data in the global frame, if an orientation estimation algorithm was used.
        Otherwise, this is None.

    Other Parameters
    ----------------
    %(other_parameters)s

    Notes
    -----
    Implementation Details:

    - If a turn has no crossing with ``lower_threshold_velocity_dps``, as it happens close to the start or the end of
      the signal, the start/ends of the turn are replaced with the start/end of the signal.
      This might not always result in accurate turn angles for these turns, and for certain cases you might want to
      remove these turns from the list.
    - The turn angle is calculated by integrating the raw unfiltered signal.

    Compared to original publication:

    - We skip the global frame transformation and assume that the z axis roughly aligns with the turning axis.
      If you want to run the turn algorithm in the global frame, perform a orientation estimation and coordinate
      system transformation before running this algorithm.

    Compared to matlab implementation:

    - We fix a bug, that only the start and the end of a turn was detected on the signed signal instead of the
      absolute signal.
      This made it "harder" to detect turns in one direction than the other.
    """

    smoothing_filter: Optional[BaseFilter]
    allowed_turn_angle_deg: tuple[float, float]
    allowed_turn_duration_s: tuple[float, float]
    min_gap_between_turns_s: float
    min_peak_angle_velocity_dps: float
    lower_threshold_velocity_dps: float
    orientation_estimation: Optional[BaseOrientationEstimation]

    global_frame_data_: Optional[pd.DataFrame]
    raw_turn_list_: pd.DataFrame
    yaw_angle_: pd.DataFrame

    def __init__(
        self,
        *,
        smoothing_filter: Optional[BaseFilter] = cf(
            ButterworthFilter(order=4, cutoff_freq_hz=0.5, filter_type="lowpass")
        ),
        min_peak_angle_velocity_dps: float = 15,
        lower_threshold_velocity_dps: float = 5,
        min_gap_between_turns_s: float = 0.05,
        allowed_turn_duration_s: tuple[float, float] = (0.5, 10),
        allowed_turn_angle_deg: tuple[float, float] = (45, np.inf),
        orientation_estimation: Optional[BaseOrientationEstimation] = None,
    ) -> None:
        self.smoothing_filter = smoothing_filter
        self.allowed_turn_angle_deg = allowed_turn_angle_deg
        self.allowed_turn_duration_s = allowed_turn_duration_s
        self.min_gap_between_turns_s = min_gap_between_turns_s
        self.min_peak_angle_velocity_dps = min_peak_angle_velocity_dps
        self.lower_threshold_velocity_dps = lower_threshold_velocity_dps
        self.orientation_estimation = orientation_estimation

    def _return_empty(self) -> Self:
        self.turn_list_ = pd.DataFrame(columns=["start", "end", "duration_s", "angle_deg", "direction"]).pipe(
            _as_valid_turn_list
        )
        self.raw_turn_list_ = self.turn_list_.copy().assign(center=[]).pipe(_as_valid_turn_list)
        return self

    @base_turning_docfiller
    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_: Unpack[dict[str, Any]]) -> Self:
        """%(detect_short)s.

        Parameters
        ----------
        %(detect_para)s

        %(detect_return)s
        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        frame = get_frame_definition(data, ["body", "global_body"])

        if self.orientation_estimation is not None:
            if frame == "global_body":
                raise ValueError(
                    "The data already seems to be in the global frame based on the available columns. "
                    "Additional orientation estimation is not possible. "
                    "Set `orientation_estimation` to None."
                )
            data = self.orientation_estimation.clone().estimate(data, sampling_rate_hz=sampling_rate_hz).rotated_data_
            gyr_is = data["gyr_gis"].to_numpy()
            self.global_frame_data_ = data
        else:
            self.global_frame_data_ = None
            is_col = "gyr_gis" if frame == "global_body" else "gyr_is"
            gyr_is = data[is_col].to_numpy()
        if self.smoothing_filter is None:
            filtered_gyr_is = gyr_is
        else:
            filtered_gyr_is = (
                self.smoothing_filter.clone().filter(gyr_is, sampling_rate_hz=sampling_rate_hz).filtered_data_
            )

        # We pre-calculate the yaw-angle here, eventhough we might not need it, if no peaks were detected.
        # This has some performance overhead, but it ensures that the yaw angle is always available as output, which
        # might be helpful for debugging, in particular when no turns were detected unexpectedly.
        yaw_angle = cumulative_trapezoid(gyr_is, dx=1 / sampling_rate_hz, initial=0)
        self.yaw_angle_ = pd.DataFrame({"angle_deg": yaw_angle}, index=data.index)

        # Note: The "abs" part here is missing in the original matlab implementation.
        abs_filtered_gyr_is = np.abs(filtered_gyr_is)
        dominant_peaks, _ = find_peaks(abs_filtered_gyr_is, height=self.min_peak_angle_velocity_dps)

        if len(dominant_peaks) == 0:
            return self._return_empty()
        # Then we find the start and the end of the turn as the first and the last sample around each peak that is above
        # the velocity threshold
        above_threshold = abs_filtered_gyr_is < self.lower_threshold_velocity_dps
        crossings = np.where(np.diff(above_threshold.astype(int)) != 0)[0]

        if len(crossings) == 0:
            return self._return_empty()

        # Vectorized search for start indices
        start_indices = np.searchsorted(crossings, dominant_peaks, side="left") - 1
        # When the index is negative (i.e. no value was before it, we set the start index to 0, aka the beginning of
        # the WB)
        starts = np.where(start_indices < 0, 0, crossings[start_indices.clip(0, len(crossings) - 1)])

        # Vectorized search for end indices
        # +1 here is required to make the end index inclusive and ensure that if for some reasone the peak is directly
        # the sample before the crossing, we still include the peak in the turn
        end_indices = np.searchsorted(crossings + 1, dominant_peaks, side="right")
        # When the index is out of bounds (i.e. no value was after it, we set the end index to the end of the WB)
        ends = np.where(
            end_indices >= len(crossings), len(data) - 1, crossings[end_indices.clip(0, len(crossings) - 1)]
        )
        # To make the end index inclusive, we add 1
        ends += 1

        def _calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
            return df.assign(
                duration_s=lambda df_: (df_["end"] - df_["start"]) / sampling_rate_hz,
                angle_deg=lambda df_: yaw_angle[df_["end"] - 1] - yaw_angle[df_["start"]],
                direction=lambda df_: np.sign(df_["angle_deg"]),
            ).replace({"direction": {1: "left", -1: "right"}})

        # Create an array of turns
        turns = pd.DataFrame({"start": starts, "end": ends}).pipe(_calculate_metrics)
        self.raw_turn_list_ = turns.copy().assign(center=dominant_peaks.astype(int)).pipe(_as_valid_turn_list)

        min_gap_turn_samples = as_samples(self.min_gap_between_turns_s, sampling_rate_hz)
        turns = (
            # For all left and all right turns, we merge turns that are closer than min_gap_between_turns_s and have the
            # same direction
            turns.groupby("direction")
            .apply(
                lambda df_: pd.DataFrame(
                    merge_intervals(df_[["start", "end"]].to_numpy(), min_gap_turn_samples),
                    columns=["start", "end"],
                ),
                include_groups=False,
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
        bool_map = turns["duration_s"].between(*self.allowed_turn_duration_s) & turns["angle_deg"].abs().between(
            *self.allowed_turn_angle_deg
        )
        self.turn_list_ = turns.loc[bool_map].copy().pipe(_as_valid_turn_list)

        return self
