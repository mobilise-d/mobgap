import numpy as np
import pandas as pd

def per_minute_counts(counts_per_sec):
    """
    Convert per-second counts to per-minute counts, including leftover seconds
    as a final partial minute.

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
    counts_per_min = counts_per_sec[:n * 60].reshape(-1, 60).sum(axis=1)

    leftover = len(counts_per_sec) % 60
    if leftover > 0:
        counts_per_min = np.append(counts_per_min, counts_per_sec[-leftover:].sum())

    return counts_per_min


def generate_weartime_list_from_minutes(weartime_flags: np.ndarray, sampling_rate: int = 100) -> pd.DataFrame:
    """
    Generate a list of wear time bouts (start and end indices) from a binary array of wear time flags per minute,
    scaling to sample indices.

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
    # Ensure input is 1D
    weartime_flags = np.asarray(weartime_flags).ravel()

    # Find change points in the binary array
    cuts = np.where(np.diff(weartime_flags) != 0)[0] + 1
    bouts = np.split(weartime_flags, cuts)

    # Start indices for each segment
    starts = [0] + cuts.tolist()

    # Keep only wear time segments and scale to samples
    wt_list = [
        (start * 60 * sampling_rate, (start + len(bout)) * 60 * sampling_rate)
        for start, bout in zip(starts, bouts)
        if bout[0] == 1
    ]

    # Convert to DataFrame
    df = pd.DataFrame(wt_list, columns=['start', 'end'])
    df.index.name = 'wt_id'

    return df

def generate_weartime_list_from_seconds(
    weartime_flags: np.ndarray,
    sampling_rate: int = 100
) -> pd.DataFrame:
    """
    Generate a list of wear-time bouts (start and end indices) from a binary
    array of wear-time flags at per-second resolution, scaled to samples.

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
    weartime_flags = np.asarray(weartime_flags).ravel()

    # Change points
    cuts = np.where(np.diff(weartime_flags) != 0)[0] + 1
    bouts = np.split(weartime_flags, cuts)
    starts = [0] + cuts.tolist()

    wt_list = [
        (start * sampling_rate,
         (start + len(bout)) * sampling_rate)
        for start, bout in zip(starts, bouts)
        if bout[0] == 1
    ]

    df = pd.DataFrame(wt_list, columns=["start", "end"])
    df.index.name = "wt_id"
    return df

def generate_weartime_list_from_samples(weartime_flags: np.ndarray) -> pd.DataFrame:
    """
    Generate wear-time bouts (start/end sample indices) from a binary array at sample resolution.

    Parameters
    ----------
    weartime_flags : np.ndarray
        1 = wear, 0 = non-wear, per sample.

    Returns
    -------
    pd.DataFrame
        Columns ['start', 'end'], index 'wt_id'.
    """
    weartime_flags = np.asarray(weartime_flags).ravel()
    cuts = np.where(np.diff(weartime_flags) != 0)[0] + 1
    starts = [0] + cuts.tolist()
    bouts = np.split(weartime_flags, cuts)

    wt_list = [
        (start, start + len(bout))
        for start, bout in zip(starts, bouts)
        if bout[0] == 1
    ]

    df = pd.DataFrame(wt_list, columns=["start", "end"])
    df.index.name = "wt_id"
    return df

def gyro_to_gyr(df: pd.DataFrame) -> pd.DataFrame:
    """Rename gyro columns to gyr to be compatible with mobgap data format."""
    return df.rename(columns={
        "gyro_x": "gyr_x",
        "gyro_y": "gyr_y",
        "gyro_z": "gyr_z",
    })
