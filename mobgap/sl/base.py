"""Base class for ICs detectors."""

from typing import Any

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc

base_sl_docfiller = make_filldoc(
    {
        "other_parameters": """
    data
        The raw IMU data of the gait sequence passed to the ``detect`` method.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz passed to the ``detect`` method.
    """,
        "slSec_list_": """
    slSec_list_
        A pandas dataframe with the values of stride length per second in the input data.
        It only has one column, ``slSec``, which contains the values of stride length per second.
    """,
        "detect_short": """
    Detect Stride length values in the passed data
    """,
        "detect_info": """
    We expect the data to be a single gait sequence with detected Initial contacts.
    If the data does not contain any gait sequences, the algorithm might behave unexpectedly.
    """,
        "detect_para": """
    data
        The raw IMU of a single sensor.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz.
    ic_list_
        The indices of the detected initial contacts in the input data.
    """,
        "detect_return": """
    Returns
    -------
    self
        The instance of the class with the ``slSec_list_`` attribute set to the estimated length per second values.
    """,
    },
    doc_summary="Decorator to fill common parts of the docstring for subclasses of :class:`BaseSlDetector`.",
)


@base_sl_docfiller
class BaseSlCalculator(Algorithm):
    """Base class for SL-calculators.

    This base class should be used for all stride length estimation algorithms.
    Algorithms should implement the ``detect`` method, which will perform all relevant processing steps.
    The method should then return the instance of the class, `slSec_list_`` attribute set to the estimated length per
    second values
    Further, the detect method should set ``self.data`` and ``self.sampling_rate_hz`` to the parameters passed to the
    method.
    We allow that subclasses specify further parameters for the detect methods (hence, this baseclass supports
    ``**kwargs``).
    However, you should only use them, if you really need them and apply active checks, that they are passed correctly.
    In 99%% of the time, you should add a new parameter to the algorithm itself, instead of adding a new parameter to
    the detect method.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(slSec_list_)s

    Notes
    -----
    You can use the :func:`~base_sl_docfiller` decorator to fill common parts of the docstring for your subclass.
    See the source of this class for an example.
    """

    _action_methods = ("calculate",)

    # Other Parameters
    data: pd.DataFrame
    sampling_rate_hz: float

    # results
    slSec_list_: pd.DataFrame

    @base_sl_docfiller
    def calculate(self, data: pd.DataFrame, *, sampling_rate_hz: float, **kwargs: Unpack[dict[str, Any]]) -> Self:
        """%(detect_short)s.

        Parameters
        ----------
        %(detect_para)s
        %(detect_return)s
        """
        raise NotImplementedError


__all__ = ["BaseSlCalculator", "base_sl_docfiller"]
