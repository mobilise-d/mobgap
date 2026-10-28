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
        The retained stride list associated with ``data``, including the stride parameters used by the calculator.
        The ``start`` and ``end`` columns, when used, are sample positions relative to this window, with end exclusive.
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
    Calculators store the inputs they consume as attributes of the same name.

    The base signature describes the shared walking-bout call: pass ``data``, ``sampling_rate_hz`` and ``stride_list``
    to every calculator in a configured collection. Subclasses explicitly declare the inputs
    they consume and accept the remaining shared inputs through ``**kwargs`` without using or storing them.
    Direct calls to a concrete calculator need only supply that calculator's required inputs. For example,
    TurnKinematics receives one prepared turn window and its sampling rate, without a stride or turn list.
    Algorithm-specific configuration belongs in initialization parameters, rather than arguments to ``calculate``.
    Subclasses may accept additional method-specific inputs through ``**kwargs``.

    Other Parameters
    ----------------
    %(data_param)s
    %(sampling_rate_param)s
    %(stride_list_param)s

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
    sampling_rate_hz: float
    stride_list: pd.DataFrame

    # results
    signal_based_parameters_: pd.Series

    perf_: MeasureTimeResults

    @base_secondary_outcome_docfiller
    def calculate(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        stride_list: pd.DataFrame,
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        """%(calculate_short)s.

        Parameters
        ----------
        %(data_param)s
        %(sampling_rate_param)s
        %(stride_list_param)s

        %(calculate_return)s
        """
        raise NotImplementedError


__all__ = ["BaseSecondaryOutcomeCalculator", "base_secondary_outcome_docfiller"]
