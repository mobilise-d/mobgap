from typing import Optional

import pandas as pd
from scipy import signal

from gaitlink.data_transform.base import BaseTransformer


class Resample(BaseTransformer):
    """
    A class for resampling input data to a specified target sampling rate using signal processing techniques.
    Derived from BaseTransformer for integration into gait analysis pipelines.


    Parameters
    ----------
    target_sampling_rate_hz : float
        The target sampling rate in Hertz.

    Attributes
    ----------
    transformed_data_ :
        The transformed data after resampling.

    data : pd.DataFrame
        The input data to be resampled.

    Methods
    -------
    transform(data, sampling_rate_hz)
        Perform the resampling action on the input data.
    """

    def __init__(self, target_sampling_rate_hz: float = 100.0) -> None:
        self.target_sampling_rate_hz = target_sampling_rate_hz
        self.transformed_data_ = pd.DataFrame()  # Initialize transformed_data_ to None

    def transform(self, data: pd.DataFrame, sampling_rate_hz: Optional[float] = None) -> "Resample":
        """
        Resample the input data to the target sampling rate.

        Parameters
        ----------
        data : pd.DataFrame
            A dataframe representing single sensor data.
        sampling_rate_hz : float
            The sampling rate of the IMU data in Hz.

        Returns
        -------
        Resample
            The instance of the transform with the results attached

        Raises
        ------
        ValueError
            If sampling_rate_hz is None or data is None.

        """
        if sampling_rate_hz is None:
            raise ValueError("Parameter 'sampling_rate_hz' must be provided.")

        if isinstance(data, pd.DataFrame):
            # Handle Pandas DataFrame

            if sampling_rate_hz == self.target_sampling_rate_hz:
                # No need to resample if the sampling rates match
                self.transformed_data_ = data  # Assign the original data
            else:

                self.data = data.copy()  # Create a copy for consistency
                # Calculate the resampling factor as a float
                resampling_factor = self.target_sampling_rate_hz / sampling_rate_hz

                resampled_data = signal.resample(data, int(len(data) * resampling_factor))

                # Create a DataFrame from the resampled data
                self.transformed_data_ = pd.DataFrame(data=resampled_data, columns=data.columns)

        else:
            # Handle other data types
            if data is None:
                raise ValueError("Parameter 'data' must be provided.")

            # Assuming data is a type that can be directly resampled (e.g., NumPy array)
            resampled_data = signal.resample(data, int(len(data) * self.target_sampling_rate_hz / sampling_rate_hz))
            self.transformed_data_ = pd.DataFrame(data=resampled_data, columns=["acc_x", "acc_y"])

        return self


