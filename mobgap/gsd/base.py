"""Base class for GSD detectors."""

from typing import Any

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc

base_gsd_docfiller = make_filldoc(
    {
        "other_parameters": """
    data
        The raw IMU data passed to the ``detect`` method.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz passed to the ``detect`` method.
    """,
        "gs_list_": """
    gs_list_
        A dataframe specifying the detected gait sequences.
        The dataframe has a ``start`` and ``end`` column, specifying the start and end index of the gait sequence.
        The values are specified as samples after the start of the recording (i.e. the start of the ``data``).
    """,
        "detect_short": """
    Detect gait sequences in the passed data
    """,
        "detect_para": """
    data
        The raw IMU of a single sensor.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz.
    """,
        "detect_return": """
    Returns
    -------
    self
        The instance of the class with the ``gs_list_`` attribute set to the detected gait sequences.
    """,
    },
    doc_summary="Decorator to fill common parts of the docstring for subclasses of :class:`BaseGsdDetector`.",
)


@base_gsd_docfiller
class BaseGsDetector(Algorithm):
    """Base class for GS-detectors.

    This base class should be used for all gait sequence detection algorithms.
    Algorithms should implement the ``detect`` method, which will perform all relevant processing steps.
    The method should then return the instance of the class, with the ``gs_list_`` attribute set to the detected
    gait sequences.
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
    %(gs_list_)s

    Notes
    -----
    You can use the :func:`~base_gsd_docfiller` decorator to fill common parts of the docstring for your subclass.
    See the source of this class for an example.

    """

    _action_methods = ("detect",)

    # Other Parameters
    data: pd.DataFrame
    sampling_rate_hz: float

    # results
    gs_list_: pd.DataFrame

    @base_gsd_docfiller
    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **kwargs: Unpack[dict[str, Any]]) -> Self:
        """%(detect_short)s.

        Parameters
        ----------
        %(detect_para)s

        %(detect_return)s
        """
        raise NotImplementedError


__all__ = ["BaseGsDetector", "base_gsd_docfiller"]
