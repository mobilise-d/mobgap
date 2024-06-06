from importlib.resources import files
from typing import Any, ClassVar, Literal, Optional, Union

import numba
import numpy as np
import pandas as pd
from scipy.signal import butter, firwin
from typing_extensions import Self, Unpack

from mobgap._docutils import inherit_docstring_from
from mobgap.data_transform._utils import chain_transformers
from mobgap.data_transform.base import (
    BaseFilter,
    FixedFilter,
    ScipyFilter,
    base_filter_docfiller,
    fixed_filter_docfiller,
    scipy_filter_docfiller,
)
from mobgap.utils.dtypes import DfLike, dflike_as_2d_array


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
        with (files("mobgap") / "data_transform/_filter_coeffs" / self._COEFFS_FILE_NAME).open() as test_data:
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

        dedrifted_data = (
            EpflDedriftFilter().filter(data, sampling_rate_hz=40.0).filtered_data_
        )
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


@numba.jit(nopython=True, cache=True)
def _hampel_filter_numba(data: np.ndarray, k: int, n_sigma: float = 3.0) -> np.ndarray:
    """Hampel filter implementation using Numba for performance optimization."""
    n = len(data)
    filtered_data = data.copy()

    gaussian_scale_factor = 1.4826

    for i in range(n):
        # Define the window range, truncating at the borders
        start = max(0, i - k)
        end = min(n, i + k + 1)

        # Compute median and MAD in the window
        window_data = data[start:end]
        median = np.nanmedian(window_data)
        mad = np.nanmedian(np.abs(window_data - median))
        sigma = gaussian_scale_factor * mad  # Scale MAD to estimate standard deviation

        # Check if the data point is an outlier
        if np.abs(data[i] - median) > n_sigma * sigma:
            filtered_data[i] = median

    return filtered_data


@base_filter_docfiller
class HampelFilter(BaseFilter):
    """Apply the Hampel filter to a time-series.

    Parameters
    ----------
    half_window_size
        The number of samples to the left and right of the current sample to use for the median filter.
        The effective window size is ``2 * half_window_size + 1`` (see ``window_size_``).
    n_sigmas
        The number of standard deviations to use for the outlier detection.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(results)s
    window_size_
        The effective window size of the filter.

    """

    half_window_size: int
    n_sigmas: float

    def __init__(self, half_window_size: int, n_sigmas: float = 3.0) -> None:
        self.half_window_size = half_window_size
        self.n_sigmas = n_sigmas

    @property
    def window_size_(self) -> int:
        """The effective window size of the filter."""
        return 2 * self.half_window_size + 1

    def filter(self, data: DfLike, *, sampling_rate_hz: Optional[float] = None, **_: Unpack[dict[str, Any]]) -> Self:
        """Apply the Hampel filter to a time-series.

        Parameters
        ----------
        data
            The series to filter.
        sampling_rate_hz
            The sampling rate of the data in Hz.
            This is ignored, as the filter does not depend on the sampling rate.

        Returns
        -------
        Self
            The instance with the filtered data attached.

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        data, index, transformation_func = dflike_as_2d_array(data)

        if data.shape[1] != 1:
            raise ValueError("The Hampel filter only supports 1-dimensional data.")

        transformed_data = _hampel_filter_numba(data.flatten(), self.half_window_size, self.n_sigmas)

        self.transformed_data_ = transformation_func(transformed_data, index)
        return self
