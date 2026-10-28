"""HarmonicRatioPeakTrough calculation for one prepared window."""

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.signal import argrelextrema
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import timed_action_method
from mobgap.secondary_outcomes.base import BaseSecondaryOutcomeCalculator, base_secondary_outcome_docfiller


@base_secondary_outcome_docfiller
class HarmonicRatioPeakTrough(BaseSecondaryOutcomeCalculator):
    """Calculate the retained peak/trough variant of harmonic ratio from acceleration.

    This variant aims to reproduce the original MATLAB harmonic-ratio approach as closely as possible by retaining
    its spectral peak/trough extraction, amplitude normalization and coefficient averaging. It is provided for
    continuity with that legacy method. Exact numerical replication of the MATLAB calculator is not guaranteed:
    Python uses the supplied stride boundaries directly rather than reconstructing or adjusting them from contacts,
    and its spectral candidate handling differs from the original implementation.

    Each supplied stride uses its own half-open start/end sample positions relative to the input signal, including
    strides from both feet. No offset subtraction, pairing of starts or endpoint adjustment is performed.
    The output names remain ``harmonic_ratio_acc_is``, ``harmonic_ratio_acc_pa`` and ``harmonic_ratio_acc_ml``.

    This estimator differs from the per-stride Fourier-amplitude definition of HR [1]_. It searches for spectral
    maxima for one harmonic group and minima for the other, averages normalized coefficients across strides,
    then divides summed even by summed odd coefficients.
    See :class:`~mobgap.secondary_outcomes.HarmonicRatioFourier` for actual harmonic
    amplitudes and averaging of individual stride ratios. That Python variant was introduced to follow the
    Fourier-amplitude definition rather than the legacy peak/trough procedure.

    Parameters
    ----------
    %(acc_columns_para)s

    Other Parameters
    ----------------
    %(data_param)s
    %(stride_list_param)s
    %(sampling_rate_param)s

    Attributes
    ----------
    %(signal_based_parameters_)s
    %(perf_)s

    Notes
    -----
    The mean-centered stride is transformed using an FFT padded to 16 times the next power of two of its length.
    The largest local maximum in 0.5-3.0 Hz for IS/PA or 0.25-1.5 Hz for ML provides amplitude normalization.
    For IS/PA, slot 2 is fixed to 1 and target frequencies are k/T. Other even slots select local maxima within
    plus/minus 1/T; odd slots select local minima in that neighborhood, ignoring minima below 0.25 Hz.
    The retained optional ML path instead uses targets 2k/T, fixes slot 1 to 1, and reverses the maxima/minima groups.
    It still returns even/odd, unlike the odd/even orientation of the Fourier method for ML.
    Missing coefficients are omitted separately when averaging each slot, and missing averaged slots are omitted
    from the sums. A zero odd sum gives NaN. A single supplied stride can contribute. No clipping is applied.
    These extraction and aggregation rules define a different statistic from conventional Fourier HR; their
    numerical results should not be treated as interchangeable. No claim of equivalent clinical validity is made.

    See Also
    --------
    mobgap.secondary_outcomes.HarmonicRatioFourier
        Harmonic ratio from Fourier amplitudes, averaged after calculating each stride's ratio.

    References
    ----------
    .. [1] J. L. Bellanca, K. A. Lowry, J. M. VanSwearingen, J. S. Brach and M. S. Redfern,
       "Harmonic ratios: A quantification of step to step symmetry," J Biomech, 46(4), 828-831, 2013.
       https://doi.org/10.1016/j.jbiomech.2012.12.008
    """

    def __init__(
        self,
        *,
        acc_columns: Optional[list[str]] = None,
    ) -> None:
        self.acc_columns = acc_columns

    @timed_action_method
    @base_secondary_outcome_docfiller
    def calculate(
        self, data: pd.DataFrame, stride_list: pd.DataFrame, sampling_rate_hz: float, **_kwargs: Unpack[dict[str, Any]]
    ) -> Self:
        """%(calculate_short)s.

        Parameters
        ----------
        %(data_param)s
        %(stride_list_param)s
        %(sampling_rate_param)s

        %(calculate_return)s
        """
        self.data = data
        self.stride_list = stride_list
        self.sampling_rate_hz = sampling_rate_hz
        self.signal_based_parameters_ = pd.Series(dtype=float)
        if stride_list is None or stride_list.empty:
            return self
        acc_columns = [c for c in self.acc_columns or [] if c in data.columns]
        hr_results = {}
        if not acc_columns:
            return self

        stride_pairs = list(stride_list[["start", "end"]].itertuples(index=False, name=None))

        for col_name in acc_columns:
            hr_val = self._process_single_accelerometer(data[col_name], stride_pairs, sampling_rate_hz)
            hr_results[f"harmonic_ratio_{col_name}"] = hr_val

        self.signal_based_parameters_ = pd.Series(hr_results, dtype=float)
        return self

    def _process_single_accelerometer(self, data: pd.Series, stride_pairs: list, sampling_rate_hz: float) -> float:
        """Process a single accelerometer axis and return the Harmonic Ratio."""
        acc = data.to_numpy()
        stride_harmonics = np.full((len(stride_pairs), 20), np.nan)
        is_ml = data.name == "acc_ml"
        gait_band = (0.25, 1.5) if is_ml else (0.5, 3.0)

        for stride_idx, (start, end) in enumerate(stride_pairs):
            stride_data = acc[start:end] - np.mean(acc[start:end])
            fft_vals, fft_freqs = self._compute_fft_spectrum(stride_data, sampling_rate_hz)

            fundamental_idx, fundamental_amp = self._find_fundamental(fft_vals, fft_freqs, gait_band)
            if fundamental_idx is None:
                continue

            # normalise by fundamental amplitude
            fft_vals /= fundamental_amp
            stride_time = len(stride_data) / sampling_rate_hz
            f1 = (2 if is_ml else 1) / stride_time

            # extract harmonic coefficients for this stride
            harmonics_this = self._extract_harmonics(fft_vals, fft_freqs, f1, is_ml)
            stride_harmonics[stride_idx, :] = harmonics_this

        # If no stride contributed, return NaN
        if np.isnan(stride_harmonics).all():
            return np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg = np.nanmean(stride_harmonics, axis=0)
            even_sum = np.nansum(avg[1::2])
            odd_sum = np.nansum(avg[0::2])
        return even_sum / odd_sum if odd_sum else np.nan

    @staticmethod
    def _compute_fft_spectrum(stride_data: np.ndarray, sampling_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute FFT magnitudes and frequencies for a stride segment."""
        n = len(stride_data)
        nfft = 2 ** (int(np.ceil(np.log2(n))) + 4)
        fft_vals = np.abs(fft(stride_data, nfft))[: nfft // 2 + 1]
        fft_freqs = np.linspace(0, sampling_rate_hz / 2, len(fft_vals))
        return fft_vals, fft_freqs

    @staticmethod
    def _find_fundamental(
        fft_vals: np.ndarray, fft_freqs: np.ndarray, gait_band: tuple
    ) -> tuple[Optional[int], Optional[float]]:
        """Find the fundamental frequency component within the gait."""
        max_idx = argrelextrema(fft_vals, np.greater)[0]
        if not max_idx.size:
            return None, None
        max_freqs = fft_freqs[max_idx]
        max_amps = fft_vals[max_idx]
        band_mask = (max_freqs >= gait_band[0]) & (max_freqs <= gait_band[1])
        if not np.any(band_mask):
            return None, None
        fundamental_idx = max_idx[band_mask][np.argmax(max_amps[band_mask])]
        fundamental_amp = fft_vals[fundamental_idx]
        return fundamental_idx, fundamental_amp

    @staticmethod
    def _extract_harmonics(fft_vals: np.ndarray, fft_freqs: np.ndarray, f1: float, is_ml: bool) -> np.ndarray:
        """Extract harmonic coefficients."""
        harmonics = f1 * np.arange(1, 21)
        stride_harmonics = np.full(20, np.nan)

        # fundamental harmonic is set to 1.0
        stride_harmonics[0 if is_ml else 1] = 1.0

        # in phase harmonics (local maxima)
        in_phase = np.arange(2, 19, 2) if is_ml else np.arange(3, 20, 2)
        max_idx = argrelextrema(fft_vals, np.greater)[0]
        if max_idx.size:
            for h in in_phase:
                f = harmonics[h]
                mask = np.abs(fft_freqs[max_idx] - f) <= f1
                if np.any(mask):
                    stride_harmonics[h] = fft_vals[max_idx[mask]].max()

        # out phase harmonics (local minima)
        out_phase = np.arange(1, 20, 2) if is_ml else np.arange(0, 19, 2)
        min_idx = argrelextrema(fft_vals, np.less)[0]
        min_idx = min_idx[fft_freqs[min_idx] >= 0.25]  # ignore very low frequencies
        if min_idx.size:
            for h in out_phase:
                f = harmonics[h]
                mask = np.abs(fft_freqs[min_idx] - f) <= f1
                if np.any(mask):
                    stride_harmonics[h] = fft_vals[min_idx[mask]].min()
        return stride_harmonics
