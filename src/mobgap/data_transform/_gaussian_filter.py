from typing import Any, Optional

from scipy.ndimage import gaussian_filter1d
from typing_extensions import Self, Unpack

from mobgap.data_transform.base import BaseFilter, base_filter_docfiller
from mobgap.utils.dtypes import DfLike, dflike_as_2d_array


@base_filter_docfiller
class GaussianFilter(BaseFilter):
    """Apply a Gaussian filter to along the time axis of a timeseries signal.

    Note that we don't use the normal FIR filter approach here, but the Gaussian filter from scipy.ndimage.
    This is done to better simulate the gaussian smoothing filter that was used in the original Matlab implementation
    of some of the algorithms in the mobilised project.

    Parameters
    ----------
    sigma_s
        Standard deviation for the Gaussian kernel in seconds.
        We use the sampling rate provided in the filter method to convert this to samples.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(results)s
    sigma_samples_
        The standard deviation for the Gaussian kernel in samples as calculated from the provided sigma_s and the
        sampling rate.

    See Also
    --------
    scipy.ndimage.gaussian_filter1d : The function that is used to apply the filter.

    """

    sigma_s: float

    sigma_samples_: float

    def __init__(self, sigma_s: float = 1.0) -> None:
        self.sigma_s = sigma_s

    @base_filter_docfiller
    def filter(self, data: DfLike, *, sampling_rate_hz: Optional[float] = None, **_: Unpack[dict[str, Any]]) -> Self:
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

        self.sigma_samples_ = self.sigma_s * sampling_rate_hz

        # Apply Gaussian filter
        self.transformed_data_ = gaussian_filter1d(
            df_data, sigma=self.sigma_samples_, axis=0, output=None, mode="reflect", cval=0.0, truncate=4.0
        )

        # Back transformation using the transformation function
        self.transformed_data_ = transformation_function(self.transformed_data_, index)

        return self
