import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, filtfilt
from gaitlink.data_transform import EpflGaitFilter

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
    fs = 200
    btype = 'high' if filter_type == 'high' else 'low'

    # adding pad
    n_coefficients = len(EpflGaitFilter().coefficients[0])
    len_pad = 4 * n_coefficients
    padded = np.pad(unfiltered_altitude, (len_pad, len_pad), "wrap")

    # ba methods gives high numbers
    #b, a = butter(order, freq / (0.5 * fs), btype=btype, analog=False, output='ba')

    # zpk methods using z and p in filtfilt gives normal
    z, p, k = butter(order, freq / (0.5 * fs), btype=btype, analog=False, output='zpk')

    # filtfilt takes two coefficients z and p give normal results
    final = filtfilt(z, p, padded)

    # Remove pad
    final = final[len_pad:-len_pad]

    return final