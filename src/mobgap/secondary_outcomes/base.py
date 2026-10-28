"""Base class for secondary mobility outcome calculations."""

from typing import Any

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc
from mobgap._utils_internal.misc import MeasureTimeResults, timer_doc_filler

base_secondary_outcome_docfiller = make_filldoc(
    {
        "signal_based_parameters_": """
    signal_based_parameters_
        A Series of scalar parameters for the supplied signal window, indexed by parameter name.
        Calculators can return multiple values. Missing required inputs can produce an empty Series;
        defined parameters with undefined numerical values retain their names and contain NaN.
    """,
        "data_param": """
    data
        The prepared IMU signal window passed to ``calculate``, with correctly oriented body-frame axes.
    """,
        "sampling_rate_param": """
    sampling_rate_hz
        The sampling rate of the IMU data in Hz passed to the ``calculate`` method.
    """,
        "stride_list_param": """
    stride_list
        The retained stride list associated with ``data``. Start/end sample positions must be relative to this window.
    """,
        "acc_columns_para": """
    acc_columns
        Name of the acceleration signal columns for which parameters will be calculated.
    """,
        "calculate_short": """
    Calculate parameters for the passed data.
    """,
        "calculate_return": """
    Returns
    -------
    self
        The instance of the class with the ``signal_based_parameters_`` attribute set to the estimated parameters.
    """,
    }
    | timer_doc_filler._dict,
    doc_summary="Decorator to fill common parts of the docstring for subclasses of "
    ":class:`~mobgap.secondary_outcomes.base.BaseSecondaryOutcomeCalculator`.",
)


@base_secondary_outcome_docfiller
class BaseSecondaryOutcomeCalculator(Algorithm):
    """Base class for secondary mobility outcome calculators.

    This base class should be used for all SDMO calculation procedures.
    Algorithms should implement the ``calculate`` method.
    The method should return the instance of the class with the ``signal_based_parameters_`` attribute.
    Every calculator consumes one prepared signal window and returns all of its scalar parameters in a Series.
    Callers select windows, preserve interval metadata and assemble results.
    Further, the calculate methods should set all inputs of the calculate method to attributes of the same name.

    Subclasses may specify further parameters for their calculate methods because individual signal-based parameters
    require different inputs.

    Other Parameters
    ----------------
    %(data_param)s
    %(sampling_rate_param)s
    %(stride_list_param)s
    replicate_matlab
        If True, use MATLAB-compatible smoothing, otherwise the direct pandas-based moving average smoothing.

    Attributes
    ----------
    %(signal_based_parameters_)s

    Notes
    -----
    Signal preparation and window selection belong to the caller. When replaying per-GS reorientation for a WB,
    check containment before applying a recorded correction; WBs need not be fully contained in a GS. Nominal frame
    conversion alone cannot correct a misoriented sensor.
    See :func:`mobgap.re_orientation.get_orientation_for_sequences`.

    You can use the :func:`~mobgap.secondary_outcomes.base.base_secondary_outcome_docfiller` decorator
    to fill common parts of the docstring for your subclass.
    See the source of this class for an example.
    """

    _action_methods = ("calculate",)

    # Other Parameters
    data: pd.DataFrame

    # results
    signal_based_parameters_: pd.Series

    perf_: MeasureTimeResults

    @base_secondary_outcome_docfiller
    def calculate(
        self,
        data: pd.DataFrame,
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        """%(calculate_short)s.

        Parameters
        ----------
        %(data_param)s

        %(calculate_return)s
        """
        raise NotImplementedError


__all__ = ["BaseSecondaryOutcomeCalculator", "base_secondary_outcome_docfiller"]
