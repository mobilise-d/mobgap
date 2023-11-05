from scipy import signal
import pandas as pd
from gaitlink.data_transform.base import BaseTransformer



class Resample(BaseTransformer):
    """
    Parameters
    ----------
    target_sampling_rate_hz : float, optional
        The target sampling rate in Hertz. Default is 100.0.

    Attributes
    ----------
    target_sampling_rate_hz : float
        The target sampling rate in Hertz.
    """
    def __init__(self, target_sampling_rate_hz=100.0):
        self.target_sampling_rate_hz = target_sampling_rate_hz
        # self.transformed_data_ = None  # Initialize transformed_data_ to None
    def transform(self, data: pd.DataFrame = None, *, sampling_rate_hz: float = None):
        """
        Resample the input data.

        Parameters
        ----------
        data : pd.DataFrame, optional
            The input data to be resampled.

        sampling_rate_hz : Target Sampling Rate, optional

        Returns
        -------
        resampled_data : pd.DataFrame
            The resampled data as a Pandas DataFrame.
        """
        if data is not None and sampling_rate_hz is not None:
            # Create a copy of the input data for consistency
            self.transformed_data_ = data.copy()

            if sampling_rate_hz == self.target_sampling_rate_hz:
                # No need to resample if the sampling rates match
                return self

            # Calculate the resampling factor as a float
            resampling_factor = self.target_sampling_rate_hz / sampling_rate_hz

            resampled_data = signal.resample(data, int((len(data) * resampling_factor)))

            # Create a DataFrame from the resampled data
            resampled_df = pd.DataFrame(data=resampled_data, columns=data.columns)

            # Update the 'transformed_data_' attribute with the resampled DataFrame
            self.transformed_data_ = resampled_df

        return self
