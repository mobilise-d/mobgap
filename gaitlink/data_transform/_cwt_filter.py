
from typing import Union

import numpy as np
from scipy.signal import cwt
from typing_extensions import Self

from gaitlink.data_transform.base import BaseTransformer
from gaitlink.utils.dtypes import DfLike, dflike_as_2d_array


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

    Other Parameters
    ----------------
    `data : np.ndarray
        The input data (assumed to be a NumPy array).
    `back_transform_func : callable or None
        Placeholder for the back-transform function.
    """

    def __init__(self, wavelet: callable, width: float) -> None:
        """
        Initialize the function.

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

    def transform(self, data: DfLike, widths: Union[list[float], None] = None, back_transform_func = None) -> Self:
        """
        Apply Continuous Wavelet Transform to analyze the input data.

        Parameters
        ----------
        data : DfLike
            Input data representing a signal.
        widths : Union[list[float], None]
            Width parameter for the transform.
        back_transform_func : callable or None, optional
            Placeholder for the back-transform function.

        Returns
        -------
        CwtFilter
            The instance of the transform with the CWT results attached.
            @param data:
            @type widths: object
        """
        if data is None:
            raise ValueError("Parameter 'data' must be provided.")
        if widths is None:
            raise ValueError("Parameter 'widths' must be provided for the transform.")

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

        except ValueError as ve:
            # Handle ValueError (e.g., invalid data or wavelet parameters)
            raise ValueError(f"Error during Continuous Wavelet Transform: {ve}")


        except TypeError:
            # Handle the case where 'dtype' is not supported by the wavelet
            self.transformed_data_ = cwt(df_data_transposed, self.wavelet, widths=widths)
            # Catching TypeError when 'dtype' is not supported by the wavelet.

        return self

    def back_transform(self) -> DfLike:
        """
        Apply a back-transform to return a copy of the original data.

        Returns
        -------
        DfLike
            A copy of the original data.
        """
        if not self.data.any():
            raise ValueError("No original data available. Perform the transformation before calling back_transform.")

        # Return a copy of the original data
        return np.copy(self.data)
