from typing import Any, Optional

from scipy.signal import savgol_filter
from typing_extensions import Self, Unpack

from mobgap.data_transform.base import BaseFilter, base_filter_docfiller
from mobgap.utils.conversions import as_samples
from mobgap.utils.dtypes import DfLike, dflike_as_2d_array


@base_filter_docfiller
class SavgolFilter(BaseFilter):
    """Apply a Savgol filter to a time series.

    .. warning:: The parameters to this filter are in seconds, not samples! We want the filter parameters to be
       independent of the sampling rate of the data and remain roughly the same frequency response, no matter the
       sampling rate.
       Therefore, the window length is provided in seconds and internally converted to samples using the sampling rate.
       The polynomial order is given in as a quantity relative to the window length.
       E.g. a window length of 0.5s and a polynonial order of 0.2 times the window length at a sampling rate of 100 Hz
       would result in an effective window length of 50 samples and a polynomial order of 10.

    Parameters
    ----------
    window_length_s
        The length of the filter window in seconds.
        This will be converted to samples using the sampling rate provided in the filter method.
    polyorder_rel
        Order of the polynomial used to fit the samples in each window.
        This is provided as a relative quantity to the window length.
        E.g. 0.5 means half the window length in samples.
        (See the warning above for more details.)

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(results)s
    window_length_samples_
        The length of the filter window in samples as calculated from the provided window length and the sampling rate.
    polyorder_
        The calculated order of the polynomial used to fit the samples in each window.

    See Also
    --------
    scipy.signal.savgol_filter : The function that is used to apply the filter.

    """

    window_length_s: float
    polyorder_rel: float

    window_length_samples_: int
    polyorder_: int

    def __init__(self, window_length_s: float, polyorder_rel: float) -> None:
        self.window_length_s = window_length_s
        self.polyorder_rel = polyorder_rel

    @base_filter_docfiller
    def filter(
        self,
        data: DfLike,
        *,
        sampling_rate_hz: Optional[float] = None,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """%(filter_short)s.

        Parameters
        ----------
        %(filter_para)s

        %(filter_return)s

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        if sampling_rate_hz is None:
            raise ValueError("Parameter 'sampling_rate_hz' must be provided.")

        # Convert to 2D array using dflike_as_2d_array function
        df_data, index, transformation_function = dflike_as_2d_array(data)

        self.window_length_samples_ = as_samples(self.window_length_s, self.sampling_rate_hz)
        self.polyorder_ = as_samples(self.window_length_s * self.polyorder_rel, self.sampling_rate_hz)

        # Apply Savitzky-Golay filter
        self.transformed_data_ = savgol_filter(
            df_data, window_length=self.window_length_samples_, polyorder=self.polyorder_, mode="mirror"
        )

        # Back transformation using the transformation function
        self.transformed_data_ = transformation_function(self.transformed_data_, index)

        return self
