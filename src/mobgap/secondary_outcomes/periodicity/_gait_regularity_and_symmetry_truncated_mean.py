"""Regularity and symmetry with a truncated centered moving average."""

import numpy as np

from mobgap.secondary_outcomes.base import base_secondary_outcome_docfiller
from mobgap.secondary_outcomes.periodicity._gait_regularity_and_symmetry import (
    GaitRegularityAndSymmetry,
    _pd_smooth_moving_ave,
)


@base_secondary_outcome_docfiller
class GaitRegularityAndSymmetryTruncatedMean(GaitRegularityAndSymmetry):
    """Calculate regularity and symmetry using a truncated centered moving average.

    This Python variant replaces the custom emulation of MATLAB's ``smooth`` operation with a standard pandas
    centered rolling mean. The change makes smoothing use the requested nominal span directly, including even
    spans, and the samples available at each boundary. This path relies on pandas' existing window implementation
    instead of a custom MATLAB smoothing routine. It changes the window convention; an accuracy advantage over
    the MATLAB convention has not been established.

    All calculations, peak-selection rules, output names and limitations are those of
    :class:`~mobgap.secondary_outcomes.GaitRegularityAndSymmetry`, except for autocovariance smoothing.
    The moving average uses the nominal span of ``max(1, floor(0.2 * sampling_rate_hz))`` samples for IS/PA
    and ``max(1, floor(0.1 * sampling_rate_hz))`` for ML. Even spans are retained. A centered pandas rolling
    mean truncates the window at the signal boundaries and averages the available samples, requiring at least one.
    These window choices are implementation choices beyond the cited regularity and symmetry definitions.

    This class and GaitRegularityAndSymmetry return the same parameter names, so use distinct algorithm keys
    when collecting both sets of results. Different smoothing can change the selected peaks and outcome values.

    Other Parameters
    ----------------
    %(data_param)s
    %(sampling_rate_param)s

    Attributes
    ----------
    %(signal_based_parameters_)s
    %(perf_)s

    See Also
    --------
    mobgap.secondary_outcomes.GaitRegularityAndSymmetry
        Regularity and symmetry with odd smoothing spans and shrinking symmetric edge windows.
    """

    @staticmethod
    def _smooth_autocovariance(values: np.ndarray, span: int) -> np.ndarray:
        return _pd_smooth_moving_ave(values, span)
