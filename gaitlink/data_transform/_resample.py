from typing import Optional

import pandas as pd
from scipy import signal

from gaitlink.data_transform.base import BaseTransformer


class Resample(BaseTransformer):
    """
    Resample the input data to the target sampling rate.

    Parameters
    ----------
    target_sampling_rate_hz : float
        The target sampling rate in Hertz.

    Attributes
    ----------
    transformed_data_ : pd.DataFrame, optional
        The resampled data as a Pandas DataFrame.

    # Other Parameters
    # ----------
    # data : pd.Dataframe, optional
    #     The data to save what we pass to the action function

    """

    def __init__(self, target_sampling_rate_hz: float = 100.0) -> None:
        self.target_sampling_rate_hz = target_sampling_rate_hz
        self.transformed_data_ = None  # Initialize transformed_data_ to None

    def transform(self, data: pd.DataFrame, sampling_rate_hz: Optional[float] = None) -> "Resample":
        """
        Resample the input data to the target sampling rate.

        Parameters
        ----------
        data : pd.Dataframe
            A dataframe representing single sensor data.
        sampling_rate_hz : float
            The sampling rate of the IMU data in Hz.

        Returns
        -------
        Resample
            The instance of the transform with the results attached

        """
        if data is not None and sampling_rate_hz is not None:
            self.data = data
            # Create a copy of the input data for consistency
            self.transformed_data_ = data.copy()

            if sampling_rate_hz == self.target_sampling_rate_hz:
                # No need to resample if the sampling rates match
                return self

            # Calculate the resampling factor as a float
            resampling_factor = self.target_sampling_rate_hz / sampling_rate_hz

            resampled_data = signal.resample(data, int(len(data) * resampling_factor))

            # Create a DataFrame from the resampled data
            resampled_df = pd.DataFrame(data=resampled_data, columns=data.columns)

            # Update the 'transformed_data_' attribute with the resampled DataFrame
            self.transformed_data_ = resampled_df

        return self
