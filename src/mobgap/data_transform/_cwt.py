from typing import Any, Optional

from pywt import cwt, frequency2scale
from typing_extensions import Self, Unpack

from mobgap.data_transform.base import BaseFilter, base_filter_docfiller
from mobgap.utils.dtypes import DfLike, dflike_as_2d_array


@base_filter_docfiller
class CwtFilter(BaseFilter):
    """Apply a Continuous Wavelet Transform (CWT) with a single fixed wavelet width as a filter.

    This acts effectively as a bandpass filter, where the scale of the wavelet determines the frequency band that is
    enhanced and the wavelet itself determines the shape of the filter.

    This means, this is not a full CWT.
    Therefore, the output has the same dimensions as the input and this could be considered a 1-1 transformation of the
    data.

    Parameters
    ----------
    wavelet
        Mother wavelet function.
        Must be set to a string of one of the wavelets supported by
        `pywt <https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html>`_.
    center_frequency_hz
        Scale of the wavelet to be used.
        This determines the frequency band the filter enhances.
        Note, that the frequency this scale corresponds to depends on the sampling rate of the data.
        You can use ``f = pywt.scale2frequency(wavelet, scale)/sampling_period`` to estimate the frequency.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(results)s
    scale_
        The wavelet scale that corresponds to the given center frequency.

    See Also
    --------
    pywt.cwt : The function that is used to apply the filter.

    """

    wavelet: str
    center_frequency_hz: float

    scale_: float

    def __init__(self, wavelet: str = "gaus2", center_frequency_hz: float = 10) -> None:
        self.wavelet = wavelet
        self.center_frequency_hz = center_frequency_hz

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
        if not isinstance(self.center_frequency_hz, (int, float)):
            raise TypeError(
                "Parameter 'width' must just be a single width for the wavelet, not a list of width as in a full CWT."
            )

        if sampling_rate_hz is None:
            raise ValueError("Parameter 'sampling_rate_hz' must be provided.")

        # Assign data without modification
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # Calculate the corresponding scale for the given frequency
        self.scale_ = frequency2scale(self.wavelet, [self.center_frequency_hz])[0] * sampling_rate_hz

        # Convert to 2D array using file_as_2d_array function from gait link.utils.dtypes
        array_2d, index, transformation_function = dflike_as_2d_array(data)

        # Apply Continuous Wavelet Transform
        output, _ = cwt(array_2d, [self.scale_], wavelet=self.wavelet, axis=0)

        self.transformed_data_ = transformation_function(output[0], index)
        return self
