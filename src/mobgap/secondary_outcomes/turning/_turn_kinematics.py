"""TurnKinematics calculation for one prepared window."""

from typing import Any

import numpy as np
import pandas as pd
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import timed_action_method
from mobgap.secondary_outcomes.base import BaseSecondaryOutcomeCalculator, base_secondary_outcome_docfiller
from mobgap.secondary_outcomes.smoothness._utils import _log_dimensionless_jerk


@base_secondary_outcome_docfiller
class TurnKinematics(BaseSecondaryOutcomeCalculator):
    """Calculate angular-velocity outcomes for one already-sliced turn window.

    The caller supplies correctly oriented body-frame data containing exactly the turn to analyze. No turn table,
    interval selection, metadata assembly, WB percentage or averaging across turns is performed here.
    ``turn_mean_ang_vel`` is abs(mean(yaw)), while ``turn_peak_ang_vel`` is max(abs(yaw)), both in degrees/s.
    ``turn_smoothness`` is dimensionless angular LDLJ over this full window, with mean-centered peak normalization.
    Negative values are expected. Duration can dominate smoothness differences between different-length turns;
    see :class:`~mobgap.secondary_outcomes.LogDimensionlessJerk`. Smoothness requires at least three samples
    and is undefined for constant yaw.
    Empty data produces an empty Series.

    Other Parameters
    ----------------
    %(data_param)s
    %(sampling_rate_param)s

    Attributes
    ----------
    %(signal_based_parameters_)s
    %(perf_)s
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
        if data.empty:
            return self
        yaw = data["gyr_is"].to_numpy()
        self.signal_based_parameters_ = pd.Series(
            {
                "turn_mean_ang_vel": np.abs(yaw.mean()),
                "turn_peak_ang_vel": np.max(np.abs(yaw)),
                "turn_smoothness": _log_dimensionless_jerk(yaw[:, None], sampling_rate_hz, derivative_order=2)[0],
            },
            dtype=float,
        )
        return self
