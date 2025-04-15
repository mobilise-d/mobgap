"""Helper functions to perform interpolation and resampling tasks."""

import warnings

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_float_dtype
from scipy.interpolate import interp1d

from mobgap.data_transform import HampelFilter
from mobgap.data_transform.base import BaseFilter


def interval_mean(
    measurement_samples: np.ndarray,
    measurements: np.ndarray,
    interval_start_ends: np.ndarray,
) -> np.ndarray:
    """Calculate the average over all measurements within the intervals.

    .. warning :: This assumes that the measurement samples are sorted!

    The method considers all measurements that are within the interval, including the start and end values.

    The calculation is performed using "nanmean", so that NaN values in the measurement are ignored.
    If no measurements are within the interval, the result will be NaN.

    Parameters
    ----------
    measurement_samples
        The samples at which the measurements were taken.
    measurements
        The measurements.
    interval_start_ends
        The start and end indices of the intervals.
        The first column ([:, 0]) contains the start indices and the second column ([:, 1]) contains the end indices.
        The unit of the indices must be the same as the unit of the measurement samples.

    Returns
    -------
    The average over all measurements within the intervals.

    """
    if len(measurement_samples) == 0:
        return np.full(len(interval_start_ends), np.nan)

    if len(interval_start_ends) == 0:
        return np.array([])

    # We use searchsorted to find the start and end indices of the intervals
    interval_start_indices = np.searchsorted(measurement_samples, interval_start_ends[:, 0], side="left")
    interval_end_indices = np.searchsorted(measurement_samples, interval_start_ends[:, 1], side="right")

    # We calculate the average over all measurements within the intervals
    # We suppress the RuntimeWarning for the case that there are no measurements within the interval
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        interval_means = np.array(
            [np.nanmean(measurements[s:e]) for s, e in zip(interval_start_indices, interval_end_indices)]
        )
    return interval_means


def robust_step_para_to_sec(
    ics_in_sec: np.ndarray,
    step_para: np.ndarray,
    sec_centers: np.ndarray,
    max_interpolation_gap_s: float,
    smoothing_filter: BaseFilter = HampelFilter(2, 3.0),
) -> np.ndarray:
    """Interpolate a per-step parameter to a per-second parameter.

    This will first smooth the per-step parameter and then interpolate it to a per-second parameter.
    The per-second parameters are then "gap-filled" by linear interpolation.
    The final per-second parameter is then smoothed again.

    Parameters
    ----------
    ics_in_sec
        The initial contacts in seconds marking the start of each step
    step_para
        The per-step parameter.
        Must have the same length as ``ics``.
        For the interpolation, we assume that the position the "step_para" is measured at is the start of the step.
    sec_centers
        The center of each second expected in the final output.
        That should be provided in samples
    max_interpolation_gap_s
        The maximum gap in seconds that is interpolated.
        If the gap is larger than this value, the second is filled with NaNs.
        We don't fill "gaps" at the start and end of the recording, as we assume that gait sequences are cut anyway
        to start and end with a valid initial contact.
    smoothing_filter
        The filter used to smooth the per-step parameter.
        The filter is applied twice, once to the raw step time and a second time on the interpolated step time values
        per second.
        We recommend to use a Hampel filter for this.

    Returns
    -------
    The per-second parameter.

    See Also
    --------
    interval_mean
        The method used to interpolate the per-step parameter to a per-second parameter.

    """
    if len(ics_in_sec) != len(step_para):
        raise ValueError("The number of initial contacts and step parameters must be equal.")

    if len(ics_in_sec) == 0:
        return np.full(len(sec_centers), np.nan)

    # 1. Smoothing
    step_time_smooth = smoothing_filter.clone().filter(step_para).transformed_data_
    # Average step time per second
    sec_start_ends = np.vstack([sec_centers - 0.5, sec_centers + 0.5]).T
    step_time_per_sec = interval_mean(ics_in_sec, step_time_smooth, sec_start_ends)
    # 2. Smoothing
    step_time_per_sec_smooth = pd.Series(smoothing_filter.filter(step_time_per_sec).transformed_data_)
    # Interpolate missing values (only inside the recording and only if the gap is smaller than the specified maximum)
    # This is not directly supported by Pandas (as the pandas ``limit`` parameter will interpolate the edges of larger
    # gaps, but can not skip larger gaps) entirely.
    # Instead, we need to segment the regions ourselves and use this as mask for the interpolation.
    # This solution is taken from https://stackoverflow.com/questions/67128364
    n_nan_mask = step_time_per_sec_smooth.notna()
    n_nan_mask = n_nan_mask.ne(n_nan_mask.shift()).cumsum()
    n_nan_mask = (
        step_time_per_sec_smooth.groupby([n_nan_mask, step_time_per_sec_smooth.isna()])
        .transform("size")
        .where(step_time_per_sec_smooth.isna())
    )
    step_time_per_sec_smooth = (
        pd.Series(step_time_per_sec_smooth)
        .interpolate(method="linear", limit_area="inside")
        .mask(n_nan_mask > max_interpolation_gap_s)
    )
    return step_time_per_sec_smooth.to_numpy()


def naive_sec_paras_to_regions(
    region_list: pd.DataFrame, sec_paras: pd.DataFrame, *, sampling_rate_hz: float
) -> pd.DataFrame:
    """Map per-second parameters to regions.

    This will map the per-second parameters to the regions specified in the region list.
    The per-second parameters are then "gap-filled" by linear interpolation.
    The final per-second parameter is then smoothed again.

    Parameters
    ----------
    region_list
        The list of regions.
        The dataframe must have a "start" and "end" column specifying the start and end indices of the regions.
        The values are specified as samples after the start of the recording (i.e. the start of the ``data``).
    sec_paras
        The per-second parameter.
        Must have the same length as the number of seconds in the recording.
        The index have a level called ``sec_center_samples`` that contains the center of the second in samples for each
        row.
    sampling_rate_hz
        The sampling rate of the recording in Hz.
        This is required to calculate the maximum extrapolation range.

    Returns
    -------
    region_list_with_paras
        The region list with all columns from ``sec_paras`` interpolated to seconds added to it.

    """
    if region_list.empty:
        return region_list.reindex(columns=[*region_list.columns, *sec_paras.columns])

    if sec_paras.empty:
        # If no second parameters are available, we return NaNs for all regions
        return pd.concat([region_list, pd.DataFrame(index=region_list.index, columns=sec_paras.columns)], axis=1)

    if non_float_dtypes := [k for k, v in sec_paras.dtypes.items() if not is_float_dtype(v)]:
        raise ValueError(
            f"The following columns of sec_paras are not of float dytpe: {non_float_dtypes}. "
            "Cast to float dtype before interpolation."
        )
    region_start_end = region_list[["start", "end"]].to_numpy().T
    sec_index = sec_paras.index.get_level_values("sec_center_samples").to_numpy()
    sec_values = sec_paras.to_numpy()
    # We use an interpolation trick here.
    # By integrating (cumsum) the values and then using linear interpolation, we can get the average
    # value per region, even if the region is not exactly aligned with the second.
    # We will basically linear interpolate between the second values.
    # For this we need to shift the second values, so that they mark the end and not the start of the second.
    sec_index = sec_index + sampling_rate_hz * 0.5
    # The problem is, that the approach does not work when we have only a single second or regions that start in the
    # second before the first value.
    # Hence, we pad the left edge.
    sec_values = np.pad(sec_values, ((1, 0), (0, 0)), mode="edge")
    sec_index = np.pad(sec_index, (1, 0), mode="constant", constant_values=(sec_index[0] - sampling_rate_hz))
    inter_vals = interp1d(
        sec_index,
        np.cumsum(sec_values, axis=0) * sampling_rate_hz,
        axis=0,
    )(region_start_end)
    # This gives us the values for all start and end values. The shape is (2, n_regions, n_columns)
    # We can now calculate the mean per region by subtracting the values at the start from the values at the end and
    # dividing by the duration of the region.
    mean_per_region = np.diff(inter_vals, axis=0)[0] / np.diff(region_start_end, axis=0)[0][:, None]

    mean_per_region = pd.DataFrame(mean_per_region, columns=sec_paras.columns, index=region_list.index).astype(
        sec_paras.dtypes
    )
    return pd.concat([region_list, mean_per_region], axis=1)
