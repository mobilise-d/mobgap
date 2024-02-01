from typing import Any, Callable, Optional

import numpy as np
from scipy.signal import cwt, ricker
from typing_extensions import Self, Unpack

from gaitlink.data_transform.base import base_filter_docfiller, BaseFilter
from gaitlink.utils.dtypes import DfLike, dflike_as_2d_array


@base_filter_docfiller
class CwtFilter(BaseFilter):
    """Apply a Continuous Wavelet Transform (CWT) with a single fixed wavelet width as a filter.

    This is not a full CWT, but we limit ourself to a single width.
    Therefore, the output has the same dimensions as the input and this could be considered a 1-1 transformation of the
    data.

    Parameters
    ----------
    wavelet
        Mother wavelet function.
    width
        Width parameter.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(results)s

    See Also
    --------
    scipy.signal.cwt : The function that is used to apply the filter.
    """

    def __init__(self, wavelet: Callable = ricker, width: float = 7) -> None:
        self.wavelet = wavelet
        self.width = width

    @base_filter_docfiller
    def filter(
        self, data: DfLike, *, sampling_rate_hz: Optional[float] = None, **kwargs: Unpack[dict[str, Any]]
    ) -> Self:
        """%(filter_short)s.

        Parameters
        ----------
        %(filter_para)s
        %(filter_kwargs)s

        %(filter_return)s

        """
        if not isinstance(self.width, (int, float)):
            raise TypeError(
                "Parameter 'width' must just be a single width for the wavelet, not a list of width as in "
                "a full CWT."
            )
        # Assign data without modification
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # Convert to 2D array using file_as_2d_array function from gait link.utils.dtypes
        df_data, index, transformation_function = dflike_as_2d_array(data)

        # Apply Continuous Wavelet Transform
        output = np.empty(df_data.shape)

        for i in range(df_data.shape[1]):
            output[:, i] = cwt(df_data[:, i], self.wavelet, widths=[self.width])

        self.transformed_data_ = transformation_function(output, index)
        return self
