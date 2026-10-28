"""Harmonic ratio from per-stride Fourier coefficients."""

from typing import Any, Optional

import numpy as np
import pandas as pd
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import timed_action_method
from mobgap.secondary_outcomes.base import BaseSecondaryOutcomeCalculator, base_secondary_outcome_docfiller


@base_secondary_outcome_docfiller
class HarmonicRatioFourier(BaseSecondaryOutcomeCalculator):
    """Calculate a Fourier-amplitude harmonic ratio for each supplied stride, then average.

    This Python variant was introduced to follow the per-stride Fourier-amplitude definition in [1]_ instead of
    reproducing the legacy MATLAB peak/trough procedure retained by
    :class:`~mobgap.secondary_outcomes.HarmonicRatioPeakTrough`. Local spectral maxima and minima need not equal
    the amplitudes at the actual harmonic frequencies, and dividing averaged coefficients need not give the mean
    of the individual stride ratios. This implementation therefore reads the harmonic amplitudes directly and
    forms each stride's ratio before averaging. The estimators can give substantially different values.

    Following the per-stride definition in [1]_, use the first 20 Fourier amplitudes at integer multiples of
    the stride frequency. IS/PA use summed even amplitudes divided by summed odd amplitudes; ML reverses the ratio.
    Each stride contributes one ratio, with equal weight, including supplied rows from both feet. The output names
    are ``harmonic_ratio_fourier_acc_is``, ``harmonic_ratio_fourier_acc_pa`` and ``harmonic_ratio_fourier_acc_ml``.
    These dimensionless outputs are separate from
    :class:`~mobgap.secondary_outcomes.HarmonicRatioPeakTrough`, which uses peak/trough extraction and
    averages normalized coefficients before calculating its ratio.

    Each stride is exactly ``data.iloc[start:end]``, with positions relative to the supplied signal and end exclusive.
    No contact-offset subtraction, pairing of starts or endpoint movement is performed. The N-point DFT represents
    the interval as one period, so its kth bin corresponds to k/T with T=N/sampling_rate_hz. The mean is removed;
    no additional filtering, tapering, zero padding or resampling is applied. This follows the harmonic definition,
    without claiming to reproduce all preprocessing or recording protocols of the reference.

    All 20 harmonics must lie strictly below Nyquist, requiring more than 40 samples per stride. Short, constant
    or nonfinite stride-axis signals give undefined ratios and are omitted from that axis's mean. If no stride
    contributes, its result is NaN. An exact zero denominator with a positive numerator gives infinity, which is
    retained in the mean; small denominators, including numerical residuals, can give very large finite values.
    No clipping or numerical/clinical QC threshold is applied. Missing or unconfigured axes are omitted. Empty
    stride lists return an empty Series. Accurate stride boundaries and adequate signal bandwidth remain important.

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

    References
    ----------
    .. [1] J. L. Bellanca, K. A. Lowry, J. M. VanSwearingen, J. S. Brach and M. S. Redfern,
       "Harmonic ratios: A quantification of step to step symmetry," J Biomech, 46(4), 828-831, 2013.
       https://doi.org/10.1016/j.jbiomech.2012.12.008
    """

    def __init__(self, *, acc_columns: Optional[list[str]] = None) -> None:
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
        results = {}
        for column in self.acc_columns or []:
            if column not in data.columns:
                continue
            signal = data[column].to_numpy(dtype=float)
            ratios = []
            for start, end in stride_list[["start", "end"]].itertuples(index=False, name=None):
                segment = signal[start:end]
                if len(segment) <= 40 or not np.isfinite(segment).all() or segment.min() == segment.max():
                    continue
                amplitudes = np.abs(np.fft.rfft(segment - segment.mean()))[1:21]
                odd, even = amplitudes[::2].sum(), amplitudes[1::2].sum()
                numerator, denominator = (odd, even) if column == "acc_ml" else (even, odd)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = numerator / denominator
                if not np.isnan(ratio):
                    ratios.append(ratio)
            results[f"harmonic_ratio_fourier_{column}"] = np.mean(ratios) if ratios else np.nan
        if results:
            self.signal_based_parameters_ = pd.Series(results, dtype=float)
        return self
