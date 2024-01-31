from typing import Any, Callable

import numpy as np
from scipy.signal import cwt, ricker
from typing_extensions import Self, Unpack
from gaitlink.data_transform.base import BaseTransformer
from gaitlink.utils.dtypes import DfLike, dflike_as_2d_array


class CwtFilter(BaseTransformer):
    """
    A class for applying Continuous Wavelet Transform (CWT) for signal analysis.

    This is not a full CWT, but we limit ourself to a single width.
    Therefore, the output has the same dimensions as the input and this could be considered a 1-1 transformation of the
    data.

    Derived from BaseTransformer for integration into gait analysis pipelines.

    Parameters
    ----------
    wavelet : Callable
        Mother wavelet function.
    width : float
        Width parameter.

    Attributes
    ----------
    transformed_data_
        The data after applying Continuous Wavelet Transform.

    Other Parameters
    ----------------
    data
        The input data

    """

    def __init__(self, wavelet: Callable = ricker, width: float = 7) -> None:
        """
        Initialize the function.

        Parameters
        ----------
        wavelet : Callable
            Mother wavelet function.
        width : float
            Width parameter.
        """

        self.wavelet = wavelet
        self.width = width

    def transform(self, data: DfLike, **_: Unpack[dict[str, Any]]) -> Self:
        """
        Apply Continuous Wavelet Transform to analyze the input data.

        Parameters
        ----------
        data : DfLike
            Input data representing a signal.

        Returns
        -------
        self
            The instance of the transform with the CWT results attached.
        """
        if not isinstance(self.width, (int, float)):
            raise TypeError("Parameter 'width' must just be a single width for the wavelet, not a list of width as in "
                            "a full CWT.")
        # Assign data without modification
        self.data = data

        # Convert to 2D array using file_as_2d_array function from gait link.utils.dtypes
        df_data, index, transformation_function = dflike_as_2d_array(data)

        # Apply Continuous Wavelet Transform
        output = np.empty(df_data.shape)

        for i in range(df_data.shape[1]):
            output[:, i] = cwt(df_data[:, i], self.wavelet, widths=[self.width])

        self.transformed_data_ = transformation_function(output, index)
        return self
