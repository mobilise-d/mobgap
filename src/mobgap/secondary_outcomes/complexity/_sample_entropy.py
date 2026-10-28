"""SampleEntropy calculation for one prepared window."""

from typing import Any, Optional

import numpy as np
import pandas as pd
from numba import njit
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import timed_action_method
from mobgap.data_transform import Resample
from mobgap.secondary_outcomes.base import BaseSecondaryOutcomeCalculator, base_secondary_outcome_docfiller


@base_secondary_outcome_docfiller
class SampleEntropy(BaseSecondaryOutcomeCalculator):
    """Calculate Sample Entropy for accelerometer data.

    The Sample Entropy is defined as the negative natural logarithm of the conditional probability
    that two sequences are similar for m points remain similar when the number of points is increased to m+1.
    Definition is taken from Torres (2013) [1]_. It can be used as a mean to describe the predictability of a
    signal.
    Usage for sway is given in Sofiane (2009) [2]_.

    Both template lengths use the common N-dim extendible starting positions. Self matches are excluded, and
    Chebyshev distance <= r times population SD defines a match. The result is -ln(A/B), where A and B count
    matching pairs of length dim+1 and dim. No B matches gives NaN; B>0 with no A matches gives positive infinity.
    Constant signals give zero with inclusive tolerance. Nonfinite axes give NaN.

    By default input is resampled to 50 Hz before calculating entropy: adjacent samples are 20 ms apart,
    and dim=2 spans 20 ms between its first and last sample, extended to 40 ms at dim+1.
    The >200 internal-sample threshold requires
    more than 4 seconds of data, not 200 steps, and is not a reliability guarantee. Fewer than two extendible
    templates give NaN even with a lowered threshold. The score is dimensionless.

    .. [1] B.D.L.C. Torres, et al. "Entropy in the Analysis of Gait Complexity: A State of the Art". British Journal
       of Applied Science & Technology. 3(4) 1097-1105, 2013.

    .. [2] R. Sofiane, et al. "On the use of sample entropy to analyze human postural sway data". Medical
       Engineering & Physics. 31, 1023-1031, 2009.

    Parameters
    ----------
    dim
        the sequence length that will be used for calculating the sample entropy. For gait dim=2 often used [1].
    r
        used for defining similarity between two sequences. Set to 0.15 as default in the original implementation.
    %(acc_columns_para)s
    num_samples_threshold
        Threshold number of samples for calculating entropy after resampling. Default is 200 from [1].
    internal_sampling_rate_hz
        Sampling rate in Hertz used internally to calculate the sample entropy. The input data is resampled to this
        sampling rate before calculating the metric.

    Other Parameters
    ----------------
    %(data_param)s
    %(sampling_rate_param)s

    Attributes
    ----------
    %(signal_based_parameters_)s
    %(perf_)s

    """

    def __init__(
        self,
        *,
        dim: int = 2,
        r: float = 0.15,
        acc_columns: Optional[list[str]] = None,
        num_samples_threshold: int = 200,
        internal_sampling_rate_hz: float = 50.0,
    ) -> None:
        self.dim = dim
        self.r = r
        self.acc_columns = acc_columns
        self.num_samples_threshold = num_samples_threshold
        self.internal_sampling_rate_hz = internal_sampling_rate_hz

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
        acc_columns = [c for c in self.acc_columns or [] if c in data.columns]
        if not acc_columns:
            return self
        dim = self.dim
        r = self.r
        num_samples = round(len(data) * self.internal_sampling_rate_hz / sampling_rate_hz)
        if num_samples <= self.num_samples_threshold or num_samples - dim < 2:
            self.signal_based_parameters_ = pd.Series(
                {f"sample_entropy_{col_name}": np.nan for col_name in acc_columns}, dtype=float
            )
            return self

        accs = data[acc_columns].astype(float)
        if sampling_rate_hz != self.internal_sampling_rate_hz:
            accs = (
                Resample(target_sampling_rate_hz=self.internal_sampling_rate_hz, attempt_index_resample=False)
                .transform(accs, sampling_rate_hz=sampling_rate_hz)
                .transformed_data_
            )

        se_results = {}
        for acc, col_name in zip(accs.to_numpy().T, acc_columns):
            value = np.nan
            if np.isfinite(acc).all():
                tol = r * np.std(acc)
                # Both lengths use the same N-dim starting positions. Pair normalization cancels.
                n_templates = len(acc) - dim
                matches_m = _count_template_matches(acc, dim, tol, n_templates)
                matches_m1 = _count_template_matches(acc, dim + 1, tol, n_templates)
                if matches_m:
                    value = -np.log(matches_m1 / matches_m) if matches_m1 else np.inf
            se_results[f"sample_entropy_{col_name}"] = value
        self.signal_based_parameters_ = pd.Series(se_results, dtype=float)
        return self


@njit
def _count_template_matches(signal: np.ndarray, m: int, tol: float, n_templates: int) -> int:
    matches = 0
    for i in range(n_templates):
        for j in range(i + 1, n_templates):
            # Chebyshev distance, inclusive tolerance, no self matches.
            for k in range(m):
                if abs(signal[i + k] - signal[j + k]) > tol:
                    break
            else:
                matches += 1
    return matches
