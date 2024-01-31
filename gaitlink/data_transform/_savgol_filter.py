import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing_extensions import Self
from gaitlink.utils.dtypes import DfLike, dflike_as_2d_array
from gaitlink.data_transform.base import BaseTransformer


class SavgolFilter(BaseTransformer):
    """
    A class for applying Savgol filter to reduce high frequency noise in a signal due to its smoothing properties

    Derived from BaseTransformer for integration into gait analysis pipelines.


    Parameters
    ----------
    window_length :
        The length of the filter window.

    polyorder :
        Order of the polynomial used to fit the samples.

    Attributes
    ----------
    transformed_data_ :
        The data after applying the Savgol filter.

    Other Parameters
    ----------------
    data
        The input data

    """

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
        self.transformed_data_ = np.array([])  # Initialize transformed_data_ to an empty DataFrame

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
        if data is None or len(data) == 0:
            raise ValueError("Parameter 'data' must be provided.")

        self.data = data

        # Convert to 2D array using dflike_as_2d_array function
        df_data, index, transformation_function = dflike_as_2d_array(data)

        # Apply Savitzky-Golay filter
        self.transformed_data_ = savgol_filter(
            df_data, window_length=self.window_length, polyorder=self.polyorder, mode="mirror"
        )

        # Back transformation using the transformation function
        self.transformed_data_ = transformation_function(self.transformed_data_, index)

        return self
