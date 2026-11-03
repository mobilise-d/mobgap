"""LogDimensionlessJerk calculation for one prepared window."""

from typing import Any, Optional

import pandas as pd
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import timed_action_method
from mobgap.secondary_outcomes.base import BaseSecondaryOutcomeCalculator, base_secondary_outcome_docfiller
from mobgap.secondary_outcomes.smoothness._utils import _log_dimensionless_jerk


@base_secondary_outcome_docfiller
class LogDimensionlessJerk(BaseSecondaryOutcomeCalculator):
    r"""Calculate the log dimensionless jerk (LDLJ) of acceleration and angular velocity signals.

    The squared jerk of each signal is integrated, normalised by the signal duration and peak amplitude to remove the
    dependence on amplitude and unit, and log-transformed [1]_. More negative values indicate less smooth signals.
    The derivative that is used depends on the type of signal:

    Acceleration signals (``acc_columns``, LDLJ-A)
        Jerk is the first derivative of acceleration:

        .. math::

            LDLJ\text{-}A = -\ln\left(\frac{T}{a_{peak}^2}\int_0^T \left(\frac{da}{dt}\right)^2 dt\right)

    Gyroscope signals (``gyr_columns``, LDLJ-V)
        The angular velocity takes the role of the velocity, so the (angular) jerk is its second derivative:

        .. math::

            LDLJ\text{-}V = -\ln\left(\frac{T^3}{\omega_{peak}^2}\int_0^T
            \left(\frac{d^2\omega}{dt^2}\right)^2 dt\right)

    with :math:`T` the duration of the signal and :math:`x_{peak} = \max |x - \bar{x}|`.
    The mean is only removed for the peak (it has no effect on the derivative).
    This removes a constant gravity projection from the peak amplitude; changing body-axis gravity projections remain.
    The result is dimensionless and independent of the unit of the signal (g or m/s², deg/s or rad/s).

    The post-processing example uses the FULL accepted walking bout, including turns and gaps between retained strides.
    Raw body axes are differentiated with finite differences; no additional filtering or gravity correction is applied.
    Results depend on input bandwidth, orientation and movement boundaries. Constant signals, fewer than three samples
    and nonfinite normalization yield NaN. Values may be negative and are dimensionless natural logarithms.

    .. warning::

        Duration can dominate smoothness differences when comparing bouts of different lengths. Repeating the same
        1 Hz motion for 5, 10 and 20 s changes acceleration LDLJ from approximately -6.20 to -7.59 to -8.97, and angular
        LDLJ from -13.09 to -15.87 to -18.64, despite unchanged per-cycle motion. Report duration alongside LDLJ.

    Parameters
    ----------
    acc_columns
        Name of the accelerometer signal columns for which LDLJ-A will be calculated.
    gyr_columns
        Name of the gyroscope signal columns for which LDLJ-V will be calculated.

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
    .. [1] A. Melendez-Calderon, C. Shirota, and S. Balasubramanian, "Estimating Movement Smoothness From Inertial
       Measurement Units," Front. Bioeng. Biotechnol., vol. 8, 2021. https://doi.org/10.3389/fbioe.2020.558771

    """

    def __init__(
        self,
        acc_columns: Optional[list[str]] = None,
        gyr_columns: Optional[list[str]] = None,
    ) -> None:
        self.acc_columns = acc_columns
        self.gyr_columns = gyr_columns

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

        out = {}
        for columns, derivative_order in ((self.acc_columns, 1), (self.gyr_columns, 2)):
            available = [column for column in columns or [] if column in data.columns]
            if not available:
                continue
            values = _log_dimensionless_jerk(data[available], sampling_rate_hz, derivative_order)
            out.update({f"jerk_{column}": value for column, value in zip(available, values)})

        if out:
            self.signal_based_parameters_ = pd.Series(out, dtype=float)
        return self
