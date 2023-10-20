from scipy import signal
import pandas as pd
from gaitlink.data_transform.base import BaseTransformer
class Resample(BaseTransformer):
    def __init__(self, target_sampling_rate_hz):
        self.target_sampling_rate_hz = target_sampling_rate_hz

    def transform(self, data: pd.DataFrame, *, sampling_rate_hz: float):
        if sampling_rate_hz == self.target_sampling_rate_hz:
            # No need to resample if the sampling rates match
            return self

        if self.target_sampling_rate_hz > sampling_rate_hz:
            # Upsample
            upsampling_factor = int(round(self.target_sampling_rate_hz / sampling_rate_hz))
            resampled_data = signal.resample(data, len(data) // upsampling_factor)

            # Create a DataFrame from the resampled data
            resampled_df = pd.DataFrame(data=resampled_data, columns=data.columns)
        else:
            # Downsample
            downsampling_factor = int(round(sampling_rate_hz / self.target_sampling_rate_hz))
            resampled_data = signal.resample(data, len(data) // downsampling_factor)

            # Create a DataFrame from the resampled data
            resampled_df = pd.DataFrame(data=resampled_data, columns=data.columns)

        # Update the 'data' attribute with the resampled DataFrame
        self.data = resampled_df
        return self
