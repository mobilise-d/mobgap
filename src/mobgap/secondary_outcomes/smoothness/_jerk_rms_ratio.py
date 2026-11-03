"""JerkRMSRatio calculation for one prepared window."""

from typing import Any

import numpy as np
import pandas as pd
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import timed_action_method
from mobgap.secondary_outcomes.base import BaseSecondaryOutcomeCalculator, base_secondary_outcome_docfiller
from mobgap.secondary_outcomes.smoothness._utils import _integrated_squared_derivative


@base_secondary_outcome_docfiller
class JerkRMSRatio(BaseSecondaryOutcomeCalculator):
    r"""Calculate the ratio of the RMS jerk of the horizontal axes to the RMS jerk of the vertical axis in dB.

    Following [1]_, the jerk is the first derivative of the acceleration, and the ratios are defined as

    .. math::

        Ratio_{ML/IS} = 10 \log_{10}\left(\frac{RMS_{ML}(jerk)}{RMS_{IS}(jerk)}\right)

    and likewise for the posterior-anterior axis (``jerk_ratio_pa_is``).
    A ratio below 0 dB means less jerk in the horizontal than in the vertical axis.
    The ratios are dimensionless and independent of the unit of the acceleration.
    Each ratio is only calculated if both of its axes are available.
    The un-normalised jerk is used on purpose: the amplitude of each axis is the information this ratio is supposed to
    capture. The amplitude normalisation of :class:`~mobgap.secondary_outcomes.LogDimensionlessJerk`
    would remove it.
    Raw body-axis acceleration is differentiated without additional filtering or gravity correction. The common RMS
    duration factor cancels in the ratio, so LDLJ's strong duration-power dependence does not apply here. Window
    composition still matters. A zero denominator or other nonfinite ratio gives NaN; fewer than three samples
    produce no values. These RMS ratios intentionally differ from differences of LDLJ scores.

    Other Parameters
    ----------------
    %(data_param)s
    %(sampling_rate_param)s

    Attributes
    ----------
    %(signal_based_parameters_)s
    %(perf_)s

    References
    ----------
    .. [1] M. A. D. Brodie, H. B. Menz, and S. R. Lord, "Age-associated changes in head jerk while walking reveal
       altered dynamic stability in older people," Exp. Brain Res., vol. 232, pp. 51-60, 2014.
       https://doi.org/10.1007/s00221-013-3719-6

    """

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

        columns = [column for column in ("acc_is", "acc_ml", "acc_pa") if column in data.columns]
        if "acc_is" not in columns or len(columns) < 2:
            return self

        acc = data[columns].to_numpy(dtype=float)
        if len(acc) < 3:
            return self
        duration = (len(acc) - 1) / sampling_rate_hz
        rms_jerk = dict(zip(columns, np.sqrt(_integrated_squared_derivative(acc, sampling_rate_hz, 1) / duration)))

        out = {}
        for axis in ("ml", "pa"):
            if f"acc_{axis}" not in rms_jerk:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_db = 10 * np.log10(rms_jerk[f"acc_{axis}"] / rms_jerk["acc_is"])
            out[f"jerk_ratio_{axis}_is"] = ratio_db if np.isfinite(ratio_db) else np.nan

        self.signal_based_parameters_ = pd.Series(out, dtype=float)
        return self
