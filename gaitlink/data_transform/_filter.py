from importlib.resources import files
from typing import Any, ClassVar, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import butter, firwin
from scipy.stats import median_abs_deviation
from typing_extensions import Self, Unpack

from gaitlink._docutils import inherit_docstring_from
from gaitlink.data_transform._utils import chain_transformers
from gaitlink.data_transform.base import (
    BaseFilter,
    FixedFilter,
    ScipyFilter,
    base_filter_docfiller,
    fixed_filter_docfiller,
    scipy_filter_docfiller,
)
from gaitlink.utils.dtypes import DfLike


@fixed_filter_docfiller
class EpflGaitFilter(FixedFilter):
    """A filter developed by EPFL to enhance gait related signals in noisy IMU data from lower-back sensors.

    .. warning::
        This filter is only intended to be used with data sampled at 40 Hz.

    Parameters
    ----------
    %(zero_phase)s

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(results)s
    %(EXPECTED_SAMPLING_RATE_HZ)s

    """

    _COEFFS_FILE_NAME: ClassVar[str] = "epfl_gait_filter.csv"
    EXPECTED_SAMPLING_RATE_HZ: ClassVar[float] = 40.0

    @property
    @inherit_docstring_from(FixedFilter)
    def coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        with (files("gaitlink") / "data_transform/_filter_coeffs" / self._COEFFS_FILE_NAME).open() as test_data:
            coeffs = pd.read_csv(test_data, header=0)["coefficients"].to_numpy()
        return coeffs, np.array(1)


@fixed_filter_docfiller
class EpflDedriftFilter(FixedFilter):
    """A custom IIR filter developed by EPFL to remove baseline drift.

    .. warning::
        This filter is only intended to be used with data sampled at 40 Hz.

    Parameters
    ----------
    %(zero_phase)s

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(results)s
    %(EXPECTED_SAMPLING_RATE_HZ)s

    """

    EXPECTED_SAMPLING_RATE_HZ: ClassVar[float] = 40.0

    @property
    @inherit_docstring_from(FixedFilter)
    def coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([1.0, -1.0]), np.array([1.0, -0.9748])


@fixed_filter_docfiller
class EpflDedriftedGaitFilter(BaseFilter):
    """A filter combining the :class:`EpflDedriftFilter` and :class:`EpflGaitFilter`.

    This filter exists, as these two filters are often used together.
    It just provides a convenient wrapper without any further optimization.
    The dedrifting filter is applied first and then the gait filter.
    I.e. it is equivalent to the following code:

    .. code-block:: python

        dedrifted_data = EpflDedriftFilter().filter(data, sampling_rate_hz=40.0).filtered_data_
        result = EpflGaitFilter().filter(dedrifted_data, sampling_rate_hz=40.0)

    .. warning::
        This filter is only intended to be used with data sampled at 40 Hz.

    Parameters
    ----------
    %(zero_phase)s

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(results)s

    """

    zero_phase: bool

    def __init__(self, zero_phase: bool = True) -> None:
        self.zero_phase = zero_phase

    @fixed_filter_docfiller
    def filter(self, data: DfLike, *, sampling_rate_hz: Optional[float] = None, **_: Unpack[dict[str, Any]]) -> Self:
        """%(filter_short)s.

        Note, that the sampling rate will not change the filter coefficients.
        Instead, the sampling rate is only used to check, that the passed data has the expected sampling rate.
        If not, a ValueError is raised.
        Hence, the ``sampling_rate_hz`` parameter only exists to make sure that you are reminded of the expected
        sampling rate.

        Parameters
        ----------
        %(filter_para)s
        %(filter_kwargs)s

        %(filter_return)s
        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        filter_chain = [
            ("dedrift", EpflDedriftFilter(zero_phase=self.zero_phase)),
            ("gait_filter", EpflGaitFilter(zero_phase=self.zero_phase)),
        ]
        self.transformed_data_ = chain_transformers(data, filter_chain, sampling_rate_hz=sampling_rate_hz)

        return self


@scipy_filter_docfiller
class ButterworthFilter(ScipyFilter):
    """Apply a butterworth filter using the transformer interface.

    Internally, this is using the :func:`scipy.signal.butter` function to design the filter coefficients using the
    second-order sections (SOS) representation.

    Parameters
    ----------
    %(common_paras)s
    %(zero_phase_sos)s

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(results)s

    """

    _FILTER_TYPE = "sos"

    def _sos_filter_design(self, sampling_rate_hz: float) -> np.ndarray:
        return butter(self.order, self.cutoff_freq_hz, btype=self.filter_type, output="sos", fs=sampling_rate_hz)


@scipy_filter_docfiller
class FirFilter(ScipyFilter):
    """Apply a fir filter using the transformer interface.

    Internally, this is using the :func:`scipy.signal.butter` function to design the filter coefficients using the
    second-order sections (SOS) representation.

    Parameters
    ----------
    %(common_paras)s
    %(zero_phase_ba)s
    window
        The window used for the FIR filter.
        This is passed to :func:`scipy.signal.firwin`.
        See `scipy.signal.get_window` for a list of windows and required parameters.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(results)s

    """

    _FILTER_TYPE = "ba"

    def __init__(
        self,
        order: int,
        cutoff_freq_hz: Union[float, tuple[float, float]],
        *,
        window: Union[str, tuple[str, Any]] = "hamming",
        filter_type: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass",
        zero_phase: bool = True,
    ) -> None:
        self.window = window
        super().__init__(order=order, cutoff_freq_hz=cutoff_freq_hz, filter_type=filter_type, zero_phase=zero_phase)

    def _ba_filter_design(self, sampling_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
        return firwin(
            self.order + 1, self.cutoff_freq_hz, pass_zero=self.filter_type, fs=sampling_rate_hz, window=self.window
        ), np.array([1])


def hampel_filter_vectorized(data: np.ndarray, window_size: int, n_sigmas: float = 3.0) -> np.ndarray:
    """Apply the Hampel filter to a time-series.

    Parameters
    ----------
    data
        The series to filter.
    window_size
        The size of the window to use for the median filter.
        Must be an odd number.
    n_sigmas
        The number of standard deviations to use for the outlier detection.

    Returns
    -------
    The filtered series.

    """
    k = 1.4826  # Scale factor for Gaussian distribution
    new_series = data.copy()

    # Create the median filtered series
    median_series = median_filter(data, size=window_size, mode="reflect")
    # Calculate the median absolute deviation with the corrected function
    scaled_mad = k * median_filter(median_abs_deviation(data, scale="normal"), size=window_size, mode="reflect")

    # Detect outliers
    outliers = np.abs(data - median_series) > n_sigmas * scaled_mad

    # Replace outliers
    new_series[outliers] = median_series[outliers]

    return new_series


@base_filter_docfiller
class HampelFilter(BaseFilter):
    """Apply the Hampel filter to a time-series.

    Parameters
    ----------
    window_size
        The size of the window to use for the median filter.
        Must be an odd number.
    n_sigmas
        The number of standard deviations to use for the outlier detection.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(results)s

    """

    window_size: int
    n_sigmas: float

    def __init__(self, window_size: int, n_sigmas: float = 3.0) -> None:
        self.window_size = window_size
        self.n_sigmas = n_sigmas

    def filter(self, data: DfLike, **_: Unpack[dict[str, Any]]) -> Self:
        """Apply the Hampel filter to a time-series.

        Parameters
        ----------
        data
            The series to filter.

        Returns
        -------
        Self
            The instance with the filtered data attached.

        """
        self.data = data
        self.transformed_data_ = hampel_filter_vectorized(data, self.window_size, self.n_sigmas)
        return self
