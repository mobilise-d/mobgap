import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def filtering_signals_100hz(input_altitude: np.ndarray, filter_type: str, freq: float, normalizing=None) -> np.ndarray:
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
    normalizing : str
        A string specifying the type of normalizing to use
        Can be 'mean' or 'mean_std'
        The default is None.

    Returns
    -------
    np.ndarray
        A numpy array containing the filtered signal

    """
    if len(input_altitude) == 0:
        return np.array([])

    filtered_altitude = np.zeros(input_altitude.shape)

    if normalizing is None:
        unfiltered_altitude = input_altitude
    #elif normalizing == 'mean_std':
        #unfiltered_altitude = normalized(input_altitude, 'mean_std')
    elif normalizing == 'mean':
        unfiltered_altitude = input_altitude - np.mean(input_altitude)
    else:
        return np.array([])
        print('Invalid normalizing type')


    if filter_type == 'high':
        filtered_altitude = apply_filter(unfiltered_altitude, freq, filter_type)

    elif filter_type == 'low':
        filtered_altitude = apply_filter(unfiltered_altitude, freq, filter_type)

    else:
        return np.array([])
        print('Invalid filter type')


    return filtered_altitude



def apply_filter(unfiltered_altitude: np.ndarray, freq: float, filter_type: str) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        A numpy array containing the filtered signal

    """
    order = 10
    btype = 'high' if filter_type == 'high' else 'low'

    b, a = butter(order, freq, btype=btype, analog=False, fs=200)
    high = filtfilt(b, a, unfiltered_altitude)
    reversed_high = high[::-1]
    high = filtfilt(b, a, reversed_high)
    high = high[::-1]

    return high
