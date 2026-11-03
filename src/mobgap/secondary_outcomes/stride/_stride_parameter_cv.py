"""StrideParameterCV calculation for one prepared window."""

import warnings
from typing import Any, Optional

import pandas as pd
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import timed_action_method
from mobgap.secondary_outcomes.base import BaseSecondaryOutcomeCalculator, base_secondary_outcome_docfiller


@base_secondary_outcome_docfiller
class StrideParameterCV(BaseSecondaryOutcomeCalculator):
    r"""Compute stride-level parameters.

    This algorithm calculates the percentage coefficient of variation in stride-level primary parameters (e.g.,
    stride length, cadence and stride duration):

    .. math::

        CV = 100*std/mean

    The standard deviation is the **sample standard deviation** (``ddof=1``, Bessel-corrected). This is not the same as
    the population standard deviation (``ddof=0``, the :func:`numpy.std` default). The two differ by a factor of
    :math:`\sqrt{n/(n-1)}`, and get closer as n grows.

    The pipeline pools retained strides from both feet. Cadence, stride length and speed are processed per-second
    estimates mapped to strides. Their CV describes variability of those mapped estimates, not direct stride-to-stride
    variability; interpolation and processing can suppress or alter it. Stride duration instead comes from same-foot
    initial-contact boundaries. Missing, spurious or misclassified contacts can heavily distort durations and their CV
    even after stride selection. For example, losing one contact can replace two 1 s intervals by one 2 s interval.
    Eight 1 s intervals and one 2 s interval give 30%% CV, although the true ten 1 s intervals have 0%% CV.

    Parameters
    ----------
    stride_list_columns
        Name of the columns in the ``stride_list`` for which parameters will be calculated.

    Other Parameters
    ----------------
    %(data_param)s
    %(stride_list_param)s

    Attributes
    ----------
    %(signal_based_parameters_)s
    %(perf_)s

    """

    def __init__(
        self,
        stride_list_columns: Optional[list[str]] = None,
    ) -> None:
        self.stride_list_columns = stride_list_columns

    @timed_action_method
    @base_secondary_outcome_docfiller
    def calculate(self, data: pd.DataFrame, stride_list: pd.DataFrame, **_kwargs: Unpack[dict[str, Any]]) -> Self:
        """%(calculate_short)s.

        Parameters
        ----------
        %(data_param)s
        %(stride_list_param)s

        %(calculate_return)s

        """
        self.data = data
        self.stride_list = stride_list
        self.signal_based_parameters_ = pd.Series(dtype=float)
        if stride_list is None or stride_list.empty or len(stride_list) < 2 or self.stride_list_columns is None:
            warnings.warn(
                "Stride-level SDMOs are not calculated because stride list is not valid.",
                stacklevel=1,
            )
            return self
        cols = self.stride_list_columns
        available_cols = [c for c in cols if c in stride_list.columns]
        if not available_cols:
            warnings.warn(
                f"Stride-level SDMOs are not calculated. None of {cols} is available in the stride list.",
                stacklevel=1,
            )
            return self
        cv = 100 * stride_list[available_cols].std() / stride_list[available_cols].mean()
        self.signal_based_parameters_ = cv.add_prefix("cv_")
        return self
