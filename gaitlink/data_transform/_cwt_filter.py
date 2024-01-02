from scipy.signal import cwt, morlet
from gaitlink.data_transform.base import BaseTransformer
import numpy as np
from gaitlink.utils.dtypes import dflike_as_2d_array
from typing_extensions import Self
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

    def __init__(self, wavelet: callable, width: float) -> None:
        """
        Parameters
        ----------
        wavelet : callable
            Mother wavelet function.
        width : float
            Width parameter.
        """
        self.wavelet = wavelet
        self.width = width
        self.transformed_data_ = np.array([])  # Initialize transformed_data_ to an empty array

    def transform(self, data) -> Self:
        """
        Apply Continuous Wavelet Transform to analyze the input data.

        Parameters
        ----------
        data : np.ndarray, pd.DataFrame, or pd.Series
            Input data representing a signal.

        Returns
        -------
        CwtFilter
            The instance of the transform with the CWT results attached.
        """
        df_data = dflike_as_2d_array(data)  # Convert to 2D array using dflike_as_2d_array function from gaitlink.utils.dtypes
        self.data = df_data  # Assign data directly without copying

        # Apply Continuous Wavelet Transform
        try:
            self.transformed_data_ = cwt(df_data, self.wavelet, widths=self.width, dtype=None)
        except TypeError:
            # Handle the case where 'method' is not supported by the wavelet
            self.transformed_data_ = cwt(df_data, self.wavelet, widths=self.width)

        return self

    def test_transform(self):
        """
        Test the transform method by comparing the output to calling the wrapped function directly.
        """
        # Test with a NumPy array
        input_data = np.array([1, 2, 3, 4, 5])
        assert np.array_equal(self.transform(input_data).transformed_data_, cwt(input_data, self.wavelet, widths=self.width))

        # Test with a pandas DataFrame
        input_df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})
        assert np.array_equal(self.transform(input_df).transformed_data_, cwt(input_df.values, self.wavelet, widths=self.width))

        # Test with a pandas Series
        input_series = pd.Series([1, 2, 3, 4, 5])
        assert np.array_equal(self.transform(input_series).transformed_data_, cwt(input_series.values, self.wavelet, widths=self.width))
