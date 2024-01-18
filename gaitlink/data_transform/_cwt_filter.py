from scipy.signal import cwt, ricker
from gaitlink.data_transform.base import BaseTransformer
import numpy as np
from gaitlink.utils.dtypes import dflike_as_2d_array, DfLike
from typing_extensions import Self
from gaitlink.data import LabExampleDataset
import matplotlib.pyplot as plt


class CwtFilter(BaseTransformer):
    """
    A class for applying Continuous Wavelet Transform (CWT) for signal analysis.
    Derived from BaseTransformer for integration into gait analysis pipelines.

    Parameters
    ----------
    wavelet : callable
        Mother wavelet function.
    width : float
        Width parameter.

    Attributes
    ----------
    transformed_data_ :
        The data after applying Continuous Wavelet Transform.

    data : np.ndarray
        The input data (assumed to be a NumPy array).
    """

    def __init__(self, wavelet, width) -> None:
        """
        Parameters
        ----------
        wavelet : callable
            Mother wavelet function.
        width : float
            Width parameter.
        """
        super().__init__()
        self.wavelet = wavelet
        self.width = width
        self.transformed_data_ = np.array([])  # Initialize transformed_data_ to an empty array
        self.back_transform_func = None  # Placeholder for the back-transform function

    def transform(self, data: DfLike, widths: list[float] = None) -> Self:
        """
        Apply Continuous Wavelet Transform to analyze the input data.

        Parameters
        ----------
        data : DfLike
            Input data representing a signal.

        Returns
        -------
        CwtFilter
            The instance of the transform with the CWT results attached.
            @param data:
            @type widths: object
        """
        if data is None:
            raise ValueError("Parameter 'data' must be provided.")

        # Assign data without modification
        self.data = data

        # Convert to 2D array using file_as_2d_array function from gait link.utils.dtypes

        df_data, _, _ = dflike_as_2d_array(data)

        if df_data.ndim == 2:
            df_data = df_data.ravel()

        df_data_transposed = df_data.T

        # Apply Continuous Wavelet Transform
        try:
            # Use a default dtype compatible with the expected output
            self.transformed_data_ = cwt(df_data_transposed, self.wavelet, widths=widths)



        except TypeError as E:
            # Handle the case where 'dtype' is not supported by the wavelet
            self.transformed_data_ = cwt(df_data_transposed, self.wavelet, widths=widths)
            # Catching TypeError when 'dtype' is not supported by the wavelet.
            pass

        return self



