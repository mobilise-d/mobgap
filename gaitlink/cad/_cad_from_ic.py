from typing import Any, Unpack

import numpy as np
import pandas as pd
from typing_extensions import Self

from gaitlink.cad.base import BaseCadenceCalculator
from gaitlink.data_transform import HampelFilter
from gaitlink.data_transform.base import BaseFilter
from gaitlink.utils.interpolation import interval_mean


def _robust_ic_to_cad_per_sec(
    ics: pd.Series, sec_centers: np.ndarray, max_interp_gap_sec: int, hampel_filter: HampelFilter
) -> pd.Series:
    ics = ics.to_numpy()
    if len(ics) < hampel_filter.window_size:
        raw_cad_per_sec = pd.Series(np.full(len(sec_centers), np.nan))
    else:
        step_time = np.diff(ics)
        # TODO: Maybe shift the step time by half a step to get the step time at the center of the interval?
        step_time_smooth = hampel_filter.filter(step_time).transformed_data_
        sec_start_ends = np.vstack([sec_centers - 0.5, sec_centers + 0.5]).T
        step_time_per_sec = interval_mean(ics[:-1], step_time_smooth, sec_start_ends)
        # We smooth the step time again to remove outliers
        step_time_per_sec_smooth = hampel_filter.filter(step_time_per_sec).transformed_data_
        step_time_per_sec_smooth = pd.Series(step_time_per_sec_smooth).interpolate(
            limit_area="inside", limit=max_interp_gap_sec
        )

        raw_cad_per_sec = 1.0 / step_time_per_sec_smooth

    return raw_cad_per_sec


class CadFromIc(BaseCadenceCalculator):
    """Calculate cadence per second directly from initial contacts.

    This uses a robust outlier removal approach to deal with missing initial contacts.
    The output cadence is reported as the average for each 1 second bin within the data.
    An incomplete second at the end is removed.

    Regions (i.e. second bouts) with no initial contacts are interpolated linearly based on the surrounding values, if
    the gap is smaller than the specified maximum interpolation gap.
    Regions without initial contacts that are larger than the specified maximum interpolation gap or at the very start
    or end of the recording are filled with NaNs.

    Parameters
    ----------
    step_time_smoothing
        The filter used to smooth the step time.
        This is used to remove outliers in the step time/cadence (e.g. when initial contacts are not detected).
        The filter is applied twice, once to the raw step time and a second time on the interpolated step time values
        per second.

    Notes
    -----
    Compared to the original Matlab implementation, this method performs all calculations on the steptime and not the
    cadence.
    The cadence values are only calculated at the very end.
    This is done to avoid potential scaling issues, as the cadence value can grow very large, when the distance
    between to initial contacts becomes very small for some reason.

    Further, we decided to use linear interpolation for gaps instead of "nearest" interpolation.
    We also don't extrapolate missing values at the start and end of the recording, but return them as NaN.

    """

    step_time_smoothing: BaseFilter
    max_interpolation_gap_s: int

    def __init__(
        self, *, step_time_smoothing: BaseFilter = HampelFilter(2, 3.0), max_interpolation_gap_s: int = 3
    ) -> None:
        self.max_interpolation_gap_s = max_interpolation_gap_s
        self.step_time_smoothing = step_time_smoothing

    def calculate(
        self,
        data: pd.DataFrame,
        initial_contacts: pd.Series,
        *,
        sampling_rate_hz: float,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """Calculate cadence directly from initial contacts.

        Parameters
        ----------
        data
            The data represented as a dataframe.
        initial_contacts
            The initial contacts represented as a series.
        sampling_rate_hz
            The sampling rate of the IMU data in Hz.

        Returns
        -------
        Self
            The instance itself.
        """
        n_secs = len(data) / sampling_rate_hz
        sec_centers = np.linspace(0, n_secs, len(data), endpoint=False) + 0.5 / sampling_rate_hz

        self.cadence_per_sec_ = _robust_ic_to_cad_per_sec(
            initial_contacts, sec_centers, self.max_interpolation_gap_s, self.step_time_smoothing.clone()
        )
        return self


# class CadFromIcDetector(BaseCadenceCalculator):
#
#     # TODO: correct typing
#     ic_detector: None
#     step_time_smoothing: HampelFilter
#
#     ic_detector_: None
#
#     def __init__(self, ic_detector, step_time_smoothing=HampelFilter(2, 3.0)):
#         self.ic_detector = ic_detector
#         self.step_time_smoothing = step_time_smoothing
#
#     @property
#     def internal_initial_contacts_(self):
#         return self.ic_detector_.initial_contacts_
#
#     def calculate(
#         self,
#         data: pd.DataFrame,
#         initial_contacts: pd.Series,
#         *,
#         sampling_rate_hz: float,
#         **kwargs: Unpack[dict[str, Any]],
#     ) -> Self:
#         """Calculate cadence directly from initial contacts.
#
#         Parameters
#         ----------
#         data
#             The data represented as a dataframe.
#         initial_contacts
#             The initial contacts represented as a series.
#             .. warning :: This method ignores the passed initial contacts and uses the internal IC detector instead.
#         sampling_rate_hz
#             The sampling rate of the IMU data in Hz.
#
#         Returns
#         -------
#         Self
#             The instance itself.
#         """
#         self.ic_detector_ = self.ic_detector.clone().detect(data, sampling_rate_hz=sampling_rate_hz)
#         new_initial_contacts = self.ic_detector_.initial_contacts_
#
#         n_secs = len(data) / sampling_rate_hz
#         sec_centers = np.linspace(0, n_secs, len(data), endpoint=False) + 0.5 / sampling_rate_hz
#
#         self.cadence_per_sec_ = _robust_ic_to_cad_per_sec(
#             new_initial_contacts.to_numpy(), sec_centers, self.step_time_smoothing.clone()
#         )
#
#         return self
