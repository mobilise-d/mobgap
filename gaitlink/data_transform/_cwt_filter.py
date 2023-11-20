from scipy.signal import cwt
from gaitlink.data_transform.base import BaseTransformer
import numpy as np

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

    Methods
    -------
    transform(data)
        Perform the CWT on the input data.
    """

    def __init__(self, wavelet: callable, width: float) -> None:
        self.wavelet = wavelet
        self.width = width
        self.transformed_data_ = np.array([])  # Initialize transformed_data_ to an empty array

    def transform(self, data: np.ndarray) -> "CwtFilter":
        """
        Apply Continuous Wavelet Transform to analyze the input data.

        Parameters
        ----------
        data : np.ndarray
            A NumPy array representing a signal.

        Returns
        -------
        CwtFilter
            The instance of the transform with the CWT results attached.

        """
        if data is None:
            raise ValueError("Parameter 'data' must be provided.")

        self.data = data.copy()  # Create a copy for consistency
        # Apply Continuous Wavelet Transform
        self.transformed_data_ = cwt(data, self.wavelet, widths=self.width, dtype=None, method='fft')

        return self
