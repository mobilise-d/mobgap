"""Base class for turn detection algorithms."""

from typing import Any

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc

base_turning_docfiller = make_filldoc(
    {
        "other_parameters": """
    data
        The raw IMU data in the body frame passed to the ``detect`` method.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz passed to the ``detect`` method.
      """,
        "turn_list_": """
    turn_list_
        The detected turns.
        The dataframe has the columns "start", "end", "duration_s", "turn_angle_deg", and "direction".
        The "direction" column specifies the direction of the turn as "left" or "right".
    """,
        "detect_short": """
        Detect turns in the passed data.
    """,
        "detect_para": """
    data
        The raw IMU data in the body frame.
        This should usually represent a single gait sequence or walking bout.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz.
    """,
        "detect_return": """
    Returns
    -------
    self
        The instance of the class with the ``turn_list_`` attribute set to the detected turns.
    """,
    },
    doc_summary="Decorator to fill common parts of the documentation for turn detection algorithms.",
)


@base_turning_docfiller
class BaseTurnDetector(Algorithm):
    """Base class for turn detection algorithms.

    This class is meant to be subclassed by turn detection algorithms.
    Algorithms should implement the ``detect`` method to detect turns in the passed data.
    The detect method should set the ``turn_list_`` attribute to the detected turns.

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
    %(turn_list_)s

    Notes
    -----
    You can use the :func:`~base_turn_docfiller` decorator to fill common parts of the docstring for your subclass.
    See the source of this class for an example.
    """

    _action_methods = ("detect",)

    # Other Parameters
    data: pd.DataFrame
    sampling_rate_hz: float

    # results
    turn_list_: pd.DataFrame

    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **kwargs: Unpack[dict[str, Any]]) -> Self:
        """%(detect_short)s.

        Parameters
        ----------
        %(detect_para)s
        %(detect_return)s
        """
        raise NotImplementedError


__all__ = ["BaseTurnDetector", "base_turning_docfiller"]
