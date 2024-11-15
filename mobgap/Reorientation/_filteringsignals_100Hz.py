import numpy as np
from typing import Literal

from mobgap.data_transform import (
    ButterworthFilter,
    chain_transformers,
)



def filtering_signals_100hz(
        input_altitude: np.ndarray,
        filter_type: str,
        freq: float,
        sampling_rate: float,
        normalizing: Literal["mean", "mean_std", None] = None
) -> np.ndarray:
    """
    Filtering the input signal with a high or low pass butterworth filter

    Parameters
    ----------
    input_altitude : np.ndarray
        A numpy array of the signal values
    filter_type : str
        A string specifying the type of filter to use
        Can be 'high' or 'low'
    freq : float
        The cutoff frequency of the filter
    sampling_rate : float
        The sampling rate of the signal
    normalizing : str
        A string specifying the type of normalizing to use
        Can be 'mean' or 'mean_std'
        The default is None.

    Returns
    -------
    np.ndarray
        A numpy array containing the filtered signal

    Raises
    ------
    ValueError
        If the normalizing is not one of the specified options.
        If the filter type is not one of the specified options.

    """
    if len(input_altitude) == 0:
        return np.array([])

    if normalizing is None:
        unfiltered_altitude = input_altitude
    elif normalizing == 'mean_std':
        raise NotImplementedError("Normalization type 'mean_std' is not implemented yet and was not used in matlab.")
    elif normalizing == 'mean':
        unfiltered_altitude = input_altitude - np.mean(input_altitude)
    else:
        raise ValueError(f"Invalid normalizing type: {normalizing}")

    if filter_type == 'high':
        filtered_altitude = apply_filter(unfiltered_altitude, freq, filter_type, sampling_rate)
    elif filter_type == 'low':
        filtered_altitude = apply_filter(unfiltered_altitude, freq, filter_type, sampling_rate)
    else:
        raise ValueError(f"Invalid filter type: {filter_type}. Must be 'high' or 'low'.")

    return filtered_altitude


def apply_filter(
        unfiltered_altitude: np.ndarray,
        freq: float,
        filter_type: str,
        sampling_rate: float
) -> np.ndarray:
    """
    Apply a high or low pass butterworth filter to the input signal
    Parameters
    ----------
    unfiltered_altitude : np.ndarray
        A numpy array of the unfiltered signal values
    freq : float
        The cutoff frequency of the filter
    filter_type : str
        A string specifying the type of filter to use
        Can be 'high' or 'low'
    sampling_rate : float
        The sampling rate of the signal

    Returns
    -------
    np.ndarray
        A numpy array containing the filtered signal

    """
    order = 10
    btype: Literal["lowpass", "highpass"] = 'highpass' if filter_type == 'high' else 'lowpass'

    filter_chain = [("butterworth", ButterworthFilter(order=order, cutoff_freq_hz=freq,
                                                      filter_type=btype, zero_phase=True))]
    final = chain_transformers(unfiltered_altitude, filter_chain, sampling_rate_hz=sampling_rate)

    return final
