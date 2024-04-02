import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gaitlink.data_transform import (
    ButterworthFilter,
    chain_transformers,
)

def filtering_signals_100hz(input_altitude: np.ndarray, filter_type: str, freq: float, sampling_rate: float, normalizing=None) -> np.ndarray:
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
        filtered_altitude = apply_filter(unfiltered_altitude, freq, filter_type, sampling_rate)

    elif filter_type == 'low':
        filtered_altitude = apply_filter(unfiltered_altitude, freq, filter_type, sampling_rate)

    else:
        return np.array([])
        print('Invalid filter type')


    return filtered_altitude



def apply_filter(unfiltered_altitude: np.ndarray, freq: float, filter_type: str, sampling_rate: float) -> np.ndarray:
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
    fs = sampling_rate
    btype = 'highpass' if filter_type == 'high' else 'lowpass'

    # plot pre
    plt.plot(unfiltered_altitude, label="unfiltered_altitude")
    plt.legend()
    plt.savefig("/Users/dimitrismegaritis/Desktop/pythonplots/unfiltered_altitude.pdf")
    plt.show()

    filter = [("butterworth", ButterworthFilter(order=order, cutoff_freq_hz=freq, filter_type=btype, zero_phase=True))]
    final = chain_transformers(unfiltered_altitude, filter, sampling_rate_hz=sampling_rate)

    # plot post
    plt.plot(final, label="final")
    plt.legend()
    plt.savefig("/Users/dimitrismegaritis/Desktop/pythonplots/final.pdf")
    plt.show()

    return final