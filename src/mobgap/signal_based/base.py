"""Base class for signal-based digital mobility outcome calculations."""

from typing import Any

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc
from mobgap._utils_internal.misc import MeasureTimeResults, timer_doc_filler

base_sdmo_docfiller = make_filldoc(
    {
        "signal_based_parameters_": """
    signal_based_parameters_
        The main output of the signal-based digital mobility outcomes (SDMO) block as a DataFrame with a single row
        and multiple columns containing the implemented signal-based parameters.
        This is a single value per metric per data.
    """,
        "data_param": """
    data
        The raw IMU data passed to the ``calculate`` method.
    """,
        "sampling_rate_param": """
    sampling_rate_hz
        The sampling rate of the IMU data in Hz passed to the ``calculate`` method.
    """,
        "stride_list_param": """
    stride_list
        The stride list associated with the ``data`` passed to the ``calculate`` method.
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
    doc_summary="Decorator to fill common parts of the docstring for subclasses of :class:`BaseSDMOCalculator`.",
)


@base_sdmo_docfiller
class BaseSDMOCalculator(Algorithm):
    """Base class for signal-based digital mobility outcome (SDMO) calculators.

    This base class should be used for all SDMO calculation procedures/classes (currently one because all
    outcomes will be calculated together as there is no need for dividing them into groups).
    Algorithms should implement the ``calculate`` method.
    The method should return the instance of the class with the ``signal_based_parameters_`` attribute.
    Further, the calculate methods should set all inputs of the calculate method to attributes of the same name.

    We allow that subclasses specify further parameters for the calculate methods (hence, this baseclass supports
    ``**kwargs``) because the signal-based parameters covers a range of different metrics, calculate methods can
    be overriding the base.

    Calculate Parameters
    ----------------
    %(data_param)s
    %(sampling_rate_param)s
    %(stride_list_param)s
    turn_list
        The turn list associated with the ``data`` passed to the ``calculate`` method.
    replicate_matlab
            If True, use MATLAB-compatible smoothing, otherwise the direct pandas-based moving average smoothing.

    Attributes
    ----------
    %(signal_based_parameters_)s

    Notes
    -----
    You can use the :func:`~base_sdmo_docfiller` decorator to fill common parts of the docstring for your subclass.
    See the source of this class for an example.
    """

    _action_methods = ("calculate",)

    # Other Parameters
    data: pd.DataFrame

    # results
    signal_based_parameters_: pd.DataFrame

    perf_: MeasureTimeResults

    @base_sdmo_docfiller
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


__all__ = ["BaseSDMOCalculator", "base_sdmo_docfiller"]
