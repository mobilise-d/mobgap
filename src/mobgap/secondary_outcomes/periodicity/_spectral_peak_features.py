"""SpectralPeakFeatures calculation for one prepared window."""

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import timed_action_method
from mobgap.secondary_outcomes.base import BaseSecondaryOutcomeCalculator, base_secondary_outcome_docfiller


@base_secondary_outcome_docfiller
class SpectralPeakFeatures(BaseSecondaryOutcomeCalculator):
    """Analyse the acceleration signal in the frequency domain.

    The amplitude and frequency were defined as the amplitude and frequency of the main peak of the power spectral
    density function. The width of the dominant PSD peak was defined as the width of the peak at half of its peak
    amplitude [1]_.

    Parameters
    ----------
    %(acc_columns_para)s
    freq_band_is
        Inclusive search band in Hz for vertical acceleration, default (0.4, 3.1).
    freq_band_ml
        Inclusive search band in Hz for mediolateral acceleration, default (0.15, 1.6).
    freq_band_pa
        Inclusive search band in Hz for anteroposterior acceleration, default (0.4, 3.1).

    Other Parameters
    ----------------
    %(data_param)s
    %(sampling_rate_param)s

    Attributes
    ----------
    %(signal_based_parameters_)s
    %(perf_)s

    Notes
    -----
    The default search bands differ from the single 0.5-3.0 Hz band specified in [1]_. They are 0.4-3.1 Hz for
    vertical and anteroposterior acceleration and 0.15-1.6 Hz for mediolateral acceleration.

    - Axes are standardized with sample SD before Welch PSD estimation. Two-second periodic Hamming windows
      (or the available shorter signal), default 50%% overlap, no further detrending and a window-based FFT length
      of twice the next power of two are retained. Total-length zero padding is also valid Welch configuration;
      zero padding changes the frequency grid, not the information resolution of the finite window.
    - The strongest local maximum within the inclusive configured band is selected, including boundary peaks
      established using neighbors outside the band. Exact ties select the lower frequency. No bin-distance pruning
      is needed to select a single dominant peak. Frequencies stay on the FFT grid. No peak gives NaN outputs.
    - Half-height crossings are linearly interpolated on both sides. Width is NaN if either crossing is unavailable
      or outside the configured band. A pure tone has nonzero width because of finite-window broadening.
    - ``amplitude_*`` is standardized-signal spectral concentration in 1/Hz, not physical acceleration amplitude.
      Frequency and width are in Hz. Constant, nonfinite or fewer-than-two-sample axes give NaN values. Missing
      or unconfigured axes are omitted; each configured anatomical axis can be calculated independently.

    References
    ----------
    .. [1] A. Weiss, E. Gazit, T. Herman, J. M. Hausdorff, and A. Mirelman,
       "Toward Automated, At-Home Assessment of Mobility Among Patients With Parkinson Disease, Using a Body-Worn
       Accelerometer," Neurorehabil Neural Repair, vol. 25, no. 9, pp. 810-818, 2011.
       https://doi.org/10.1177/1545968311424869

    .. [2] D. Pradeep Kumar, N. Toosizadeh, J. Mohler, H. Ehsani, C. Mannier, and K. Laksari,
       "Sensor-based characterization of daily walking: a new paradigm in pre-frailty/frailty assessment,"
       BMC Geriatr, vol. 20, no. 1, p. 164, 2020.

    """

    def __init__(
        self,
        acc_columns: Optional[list[str]] = None,
        freq_band_is: tuple[float, float] = (0.4, 3.1),
        freq_band_ml: tuple[float, float] = (0.15, 1.6),
        freq_band_pa: tuple[float, float] = (0.4, 3.1),
    ) -> None:
        self.acc_columns = acc_columns
        self.freq_band_is = freq_band_is
        self.freq_band_ml = freq_band_ml
        self.freq_band_pa = freq_band_pa

    @timed_action_method
    @base_secondary_outcome_docfiller
    def calculate(self, data: pd.DataFrame, sampling_rate_hz: float, **_kwargs: Unpack[dict[str, Any]]) -> Self:
        """%(calculate_short)s.

        Parameters
        ----------
        %(data_param)s
        %(sampling_rate_param)s

        %(calculate_return)s

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.signal_based_parameters_ = pd.Series(dtype=float)
        results = {}
        for axis, band in (("is", self.freq_band_is), ("ml", self.freq_band_ml), ("pa", self.freq_band_pa)):
            column = f"acc_{axis}"
            if column not in (self.acc_columns or []) or column not in data.columns:
                continue
            values = data[column]
            amplitude, frequency, width = np.nan, np.nan, np.nan
            if len(values) >= 2 and np.isfinite(values).all() and values.min() != values.max() and values.std() > 0:
                values = (values - values.mean()) / values.std()
                win_size = min(int(sampling_rate_hz * 2), len(values))
                fft_length = 2 ** (int(np.ceil(np.log2(win_size))) + 1)
                freq, psd = welch(
                    values.to_numpy(),
                    fs=sampling_rate_hz,
                    window="hamming",
                    nperseg=win_size,
                    nfft=fft_length,
                    detrend=False,
                )
                amplitude, frequency, width = self._extract_amp_freq_width(psd, freq, band)
            results.update({f"amplitude_{axis}": amplitude, f"freq_{axis}": frequency, f"width_{axis}": width})
        if results:
            self.signal_based_parameters_ = pd.Series(results, dtype=float)
        return self

    @staticmethod
    def _extract_amp_freq_width(
        psd: np.ndarray, freq: np.ndarray, band: tuple[float, float]
    ) -> tuple[float, float, float]:
        """Select the strongest in-band local peak and interpolate its half-height crossings."""
        peaks, _ = find_peaks(psd)
        lower, upper = band
        peaks = peaks[(freq[peaks] >= lower) & (freq[peaks] <= upper)]
        if not len(peaks):
            return np.nan, np.nan, np.nan
        peak = peaks[np.argmax(psd[peaks])]
        amplitude = psd[peak]
        half_height = amplitude / 2
        left = np.flatnonzero(psd[:peak] <= half_height)
        right = np.flatnonzero(psd[peak + 1 :] <= half_height)
        if not len(left) or not len(right):
            return amplitude, freq[peak], np.nan
        i = left[-1]
        j = peak + 1 + right[0]
        left_cross = freq[i] + (half_height - psd[i]) / (psd[i + 1] - psd[i]) * (freq[i + 1] - freq[i])
        right_cross = freq[j - 1] + (half_height - psd[j - 1]) / (psd[j] - psd[j - 1]) * (freq[j] - freq[j - 1])
        width = right_cross - left_cross if lower <= left_cross and right_cross <= upper else np.nan
        return amplitude, freq[peak], width
