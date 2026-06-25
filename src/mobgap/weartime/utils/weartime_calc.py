"""Functions for calculating wear-time statistics and conversions."""

import numpy as np
import pandas as pd

from mobgap.weartime.utils._intervals import flags_to_intervals, intervals_to_weartime_df


def per_minute_counts(counts_per_sec: np.ndarray) -> np.ndarray:
    """
    Convert per-second counts to per-minute counts.

    Includes leftover seconds as a final partial minute.

    Parameters
    ----------
    counts_per_sec : np.ndarray
        1D array of per-second activity counts.

    Returns
    -------
    counts_per_min : np.ndarray
        1D array of per-minute counts. Last element may be less than 60 s.
    """
    counts_per_sec = np.asarray(counts_per_sec)

    n = len(counts_per_sec) // 60
    counts_per_min = counts_per_sec[: n * 60].reshape(-1, 60).sum(axis=1)

    leftover = len(counts_per_sec) % 60
    if leftover > 0:
        counts_per_min = np.append(counts_per_min, counts_per_sec[-leftover:].sum())

    return counts_per_min


def generate_weartime_list_from_minutes(weartime_flags: np.ndarray, sampling_rate: int = 100) -> pd.DataFrame:
    """
    Generate a list of wear time bouts from binary flags per minute.

    Scales to sample indices.

    Parameters
    ----------
    weartime_flags : np.ndarray
        Binary array (1 = wear time, 0 = non-wear time) at the per-minute level.
    sampling_rate : int, optional
        Number of samples per minute, e.g., 60 for per-second resolution. Default is 60.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['start', 'end'] and index as 'wt_id' representing wear time bouts in samples.
    """
    intervals = flags_to_intervals(weartime_flags) * (60 * sampling_rate)
    return intervals_to_weartime_df(intervals)


def generate_weartime_list_from_seconds(weartime_flags: np.ndarray, sampling_rate: int = 100) -> pd.DataFrame:
    """
    Generate a list of wear-time bouts from binary flags at per-second resolution.

    Scaled to samples.

    Parameters
    ----------
    weartime_flags : np.ndarray
        Binary array (1 = wear, 0 = non-wear) at per-second resolution.
    sampling_rate : int
        Sampling rate in Hz.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['start', 'end'] (sample indices),
        indexed by 'wt_id'.
    """
    intervals = flags_to_intervals(weartime_flags) * sampling_rate
    return intervals_to_weartime_df(intervals)


def generate_weartime_list_from_samples(weartime_flags: np.ndarray) -> pd.DataFrame:
    """
    Generate wear-time bouts from binary array at sample resolution.

    Parameters
    ----------
    weartime_flags : np.ndarray
        1 = wear, 0 = non-wear, per sample.

    Returns
    -------
    pd.DataFrame
        Columns ['start', 'end'], index 'wt_id'.
    """
    return intervals_to_weartime_df(flags_to_intervals(weartime_flags))


def gyro_to_gyr(df: pd.DataFrame) -> pd.DataFrame:
    """Rename gyro columns to gyr to be compatible with mobgap data format."""
    return df.rename(
        columns={
            "gyro_x": "gyr_x",
            "gyro_y": "gyr_y",
            "gyro_z": "gyr_z",
        }
    )
