from typing import Any, Optional, Unpack

from scipy.ndimage import gaussian_filter1d
from typing_extensions import Self

from gaitlink.data_transform.base import BaseFilter, base_filter_docfiller
from gaitlink.utils.dtypes import DfLike, dflike_as_2d_array


@base_filter_docfiller
class GaussianFilter(BaseFilter):
    """Apply a Gaussian filter to along the time axis of a timeseries signal.

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

    See Also
    --------
    scipy.ndimage.gaussian_filter1d : The function that is used to apply the filter.

    """

    sigma_s: float

    def __init__(self, sigma_s: float = 1.0) -> None:
        self.sigma_s = sigma_s

    @base_filter_docfiller
    def filter(
        self,
        data: DfLike,
        *,
        sampling_rate_hz: Optional[float] = None,
        **kwargs: Unpack[dict[str, Any]],  # noqa: ARG002
    ) -> Self:
        """%(filter_short)s.

        Parameters
        ----------
        %(filter_para)s
        %(filter_kwargs)s

        %(filter_return)s

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # Convert to 2D array using dflike_as_2d_array function
        df_data, index, transformation_function = dflike_as_2d_array(data)

        sigma_samples = self.sigma_s * sampling_rate_hz

        # Apply Gaussian filter
        self.transformed_data_ = gaussian_filter1d(
            df_data, sigma=sigma_samples, axis=0, output=None, mode="reflect", cval=0.0, truncate=4.0
        )

        # Back transformation using the transformation function
        self.transformed_data_ = transformation_function(self.transformed_data_, index)

        return self
