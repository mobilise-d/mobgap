from typing import Any, Optional

from scipy.signal import savgol_filter
from typing_extensions import Self, Unpack

from gaitlink.data_transform.base import BaseFilter, base_filter_docfiller
from gaitlink.utils.dtypes import DfLike, dflike_as_2d_array


@base_filter_docfiller
class SavgolFilter(BaseFilter):
    """Apply a Savgol filter to a time series.

    Parameters
    ----------
    window_length
        The length of the filter window.
    polyorder
        Order of the polynomial used to fit the samples.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(results)s

    See Also
    --------
    scipy.signal.savgol_filter : The function that is used to apply the filter.

    """

    def __init__(self, window_length: int = 5, polyorder: int = 2) -> None:
        self.window_length = window_length
        self.polyorder = polyorder

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
