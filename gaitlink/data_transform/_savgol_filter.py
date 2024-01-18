import pandas as pd
from scipy.signal import savgol_filter
from typing_extensions import Self
from gaitlink.data_transform.base import BaseTransformer


class SavgolFilter(BaseTransformer):
    transformed_data_: pd.DataFrame  # Class-level type hint for transformed_data_

    def __init__(self, window_length: int = 5, polyorder: int = 2) -> None:
        """
        Initialize the SavgolFilter with specified parameters.

        Parameters
        ----------
        window_length : int, optional
            The length of the filter window, by default 5.
        polyorder : int, optional
            The order of the polynomial to fit, by default 2.
        """
        self.window_length = window_length
        self.polyorder = polyorder
        self.transformed_data_ = pd.DataFrame()  # Initialize transformed_data_ to an empty DataFrame

    def transform(self, data: pd.DataFrame) -> Self:
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
        self.transformed_data_ = savgol_filter(data, window_length=self.window_length, polyorder=self.polyorder, mode='mirror')

        return self
