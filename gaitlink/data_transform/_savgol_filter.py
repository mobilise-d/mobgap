from typing import Optional

import pandas as pd
from scipy.signal import savgol_filter

from gaitlink.data_transform.base import BaseTransformer


class SavgolFilter(BaseTransformer):
    """
    A class for applying Savitzky-Golay filter to smoothen noisy data.
    Derived from BaseTransformer for integration into gait analysis pipelines.


    Parameters
    ----------
    window_length : int
        The length of the filter window.
    polyorder : int
        The order of the polynomial used to fit the samples.

    Attributes
    ----------
    transformed_data_ :
        The smoothed data after applying the Savitzky-Golay filter.

    data : pd.DataFrame
        The input data to be smoothed.

    Methods
    -------
    transform(data)
        Perform the smoothing action on the input data.
    """

    def __init__(self, window_length: int = 5, polyorder: int = 2) -> None:
        self.window_length = window_length
        self.polyorder = polyorder
        self.transformed_data_ = pd.DataFrame()  # Initialize transformed_data_ to None

    def transform(self, data: pd.DataFrame) -> "SavgolFilter":
        """
        Apply the Savitzky-Golay filter to smoothen the input data.

        Parameters
        ----------
        data : pd.DataFrame
            A dataframe representing single sensor data.

        Returns
        -------
        SavgolFilter
            The instance of the transform with the smoothed results attached.

        """
        if data is None:
            raise ValueError("Parameter 'data' must be provided.")

        self.data = data.copy()  # Create a copy for consistency
        # Apply Savitzky-Golay filter
        self.transformed_data_ = savgol_filter(data, window_length=self.window_length, polyorder=self.polyorder, axis=0)

        return self
