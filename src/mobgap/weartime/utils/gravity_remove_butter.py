import numpy as np
import pandas as pd
from mobgap.data_transform import (
    chain_transformers,
    ButterworthFilter
)

def gravity_motion_butterworth(data: pd.DataFrame, sampling_rate_hz: float):
    """
    Separate linear acceleration (motion) from the gravity component using
    a low-pass Butterworth filter.

    This function applies a first-order Butterworth low-pass filter
    (cutoff frequency = 0.25 Hz) to the input accelerometer signals to
    estimate the gravity component. The motion component is obtained by
    subtracting the gravity component from the raw signal.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe containing raw accelerometer signals with the following columns:
        - 'acc_is' : Acceleration along the **inferior–superior (IS)** axis.
        - 'acc_ml' : Acceleration along the **medio–lateral (ML)** axis.
        - 'acc_pa' : Acceleration along the **posterior–anterior (PA)** axis.
    sampling_rate_hz : float
        Sampling frequency of the signals in Hertz (Hz).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the motion component (gravity removed) for each axis:
        - 'acc_is' : Motion along IS axis.
        - 'acc_ml' : Motion along ML axis.
        - 'acc_pa' : Motion along PA axis.

    Raises
    ------
    ValueError
        If one or more required columns ('acc_is', 'acc_ml', 'acc_pa')
        are missing from the input DataFrame.

    Notes
    -----
    - The Butterworth filter is implemented with order=1 and cutoff=0.25 Hz,
      following the reference used in the original methodology.
    - Gravity is assumed to dominate frequencies below 0.25 Hz.
    """

    # Ensuring required keys exist in data
    required_keys = ['acc_is', 'acc_ml', 'acc_pa']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: '{key}' in input data")

    acc_is = data['acc_is'].values
    acc_ml = data['acc_ml'].values
    acc_pa = data['acc_pa'].values

    # Performing a low pass butterworth filter on the data
    cutoff = 0.25
    # class instance
    filter_chain = [("butter", ButterworthFilter(order=1, cutoff_freq_hz=cutoff, filter_type='lowpass'))]

    # application to all corrected axes
    acc_is_filt = np.asarray(chain_transformers(acc_is, filter_chain, sampling_rate_hz=sampling_rate_hz))
    acc_ml_filt = np.asarray(chain_transformers(acc_ml, filter_chain, sampling_rate_hz=sampling_rate_hz))
    acc_pa_filt = np.asarray(chain_transformers(acc_pa, filter_chain, sampling_rate_hz=sampling_rate_hz))


    # Compute motion components by subtracting gravity
    acc_is_no_grav = data['acc_is'] - acc_is_filt
    acc_ml_no_grav = data['acc_ml'] - acc_ml_filt
    acc_pa_no_grav = data['acc_pa'] - acc_pa_filt

    # concatinate the data
    acc_no_grav = pd.concat([acc_is_no_grav, acc_ml_no_grav, acc_pa_no_grav], axis=1)

    return acc_no_grav