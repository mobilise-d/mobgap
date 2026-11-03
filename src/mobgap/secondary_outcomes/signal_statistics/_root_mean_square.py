"""RootMeanSquare calculation for one prepared window."""

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import timed_action_method
from mobgap.secondary_outcomes.base import BaseSecondaryOutcomeCalculator, base_secondary_outcome_docfiller


@base_secondary_outcome_docfiller
class RootMeanSquare(BaseSecondaryOutcomeCalculator):
    """Compute root-mean-square (RMS) parameters for acceleration and gyroscope signals.

    Acceleration signals are mean-centered before calculating their per-axis RMS. The total acceleration RMS is the
    Euclidean norm of the per-axis acceleration RMS values, and each acceleration ratio is the corresponding per-axis
    RMS divided by that total. Gyroscope signals are not mean-centered and are reported as independent per-axis RMS
    parameters; they do not contribute to the total acceleration RMS or acceleration ratios.

    Total RMS requires all three anatomical axes (``acc_is``, ``acc_ml``, ``acc_pa``) to be configured and present.
    Otherwise the total and ratios are NaN; configured per-axis RMS values remain available. Extra configured
    channels never contribute to the anatomical total. Zero-total ratios are also NaN. RMS divides by N,
    acceleration is in m/s² and angular velocity in degrees/s. There is no unit conversion or linear detrending.

    The acceleration RMS ratios follow:
    A gait abnormality measure based on root mean square of trunk acceleration.
    Masaki Sekine et al. Journal of NeuroEngineering and Rehabilitation 2013, 10:118.
    https://doi.org/10.1186/1743-0003-10-118

    Parameters
    ----------
    %(acc_columns_para)s
    gyr_columns
        Name of the gyroscope columns to compute per-axis RMS for.

    Other Parameters
    ----------------
    %(data_param)s

    Attributes
    ----------
    %(signal_based_parameters_)s
    %(perf_)s

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
    def calculate(self, data: pd.DataFrame, **_kwargs: Unpack[dict[str, Any]]) -> Self:
        """%(calculate_short)s.

        Parameters
        ----------
        %(data_param)s

        %(calculate_return)s
        """
        self.data = data
        self.signal_based_parameters_ = pd.Series(dtype=float)
        acc_columns = [c for c in (self.acc_columns or []) if c in data.columns]
        gyr_columns = [c for c in (self.gyr_columns or []) if c in data.columns]
        if not acc_columns and not gyr_columns:
            warnings.warn(
                "No acceleration or gyroscope columns are available.",
                stacklevel=1,
            )
            return self

        signals = data[[*acc_columns, *gyr_columns]].astype(float)
        # Remove the DC component of the acceleration signals.
        if acc_columns:
            # A preliminary constant shift makes constant floating-point axes exactly zero. This is still
            # mean removal, (x-c)-mean(x-c) == x-mean(x), without a numerical threshold for zero RMS.
            signals[acc_columns] = signals[acc_columns].subtract(signals[acc_columns].min(), axis=1)
            signals[acc_columns] = signals[acc_columns].subtract(signals[acc_columns].mean(), axis=1)

        rms = np.sqrt(signals.pow(2).mean()).add_prefix("rms_")
        if acc_columns:
            acc_rms_columns = [f"rms_{column}" for column in acc_columns]
            rms_total_acc = (
                np.linalg.norm(rms[["rms_acc_is", "rms_acc_ml", "rms_acc_pa"]])
                if {"acc_is", "acc_ml", "acc_pa"}.issubset(acc_columns)
                else np.nan
            )
            rms["rms_total_acc"] = rms_total_acc
            for column, rms_column in zip(acc_columns, acc_rms_columns):
                rms[f"rms_ratio_{column}"] = rms[rms_column] / rms_total_acc if rms_total_acc != 0 else np.nan
        self.signal_based_parameters_ = rms
        return self
