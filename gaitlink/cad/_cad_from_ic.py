import numpy as np

from gaitlink.cad.base import BaseCadenceCalculator
from gaitlink.utils._filter import hampel_filter_vectorized
from gaitlink.utils.interpolation import interval_mean


def _robust_ic_to_cad_per_sec(ics: np.ndarray, sec_centers: np.ndarray, *, hampel_filter_window_size: int =5, hampel_n_sigmas: float = 3.0):
    if len(ics) < 2:
        raw_cad_per_sec = np.full(len(sec_centers), np.nan)
    else:
        step_time = np.diff(ics)
        # TODO: Maybe shift the step time by half a step to get the step time at the center of the interval?
        step_time_smooth = hampel_filter_vectorized(step_time, hampel_filter_window_size, hampel_n_sigmas)
        sec_start_ends = np.vstack([sec_centers - 0.5, sec_centers + 0.5]).T
        step_time_per_sec = interval_mean(ics[:-1], step_time_smooth, sec_start_ends)


class DirectCadFromIc(BaseCadenceCalculator):
    """Calculate cadence directly from initial contacts.

    This uses a robust outlier removal approach to deal with missing initial contacts.
    """

    def