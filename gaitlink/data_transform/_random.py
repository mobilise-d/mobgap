from scipy import signal
import pandas as pd
from gaitlink.data_transform.base import BaseTransformer



class Resample(BaseTransformer):
    def __init__(self, target_sampling_rate_hz):
        self.target_sampling_rate_hz = target_sampling_rate_hz
        self.transformed_data_ = None

    def transform(self, data: pd.DataFrame, *, sampling_rate_hz: float):
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