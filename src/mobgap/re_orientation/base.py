"""Base class for reorientation correction algorithms."""

from typing import Any

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc

base_reorientation_docfiller = make_filldoc(
    {
        "other_parameters": """
    data
        The raw IMU data in the body frame passed to the ``detect_correct`` method.
    """,
        "corrected_data_": """
    corrected_data_
        The reoriented IMU data in the anatomical frame.
        The dataframe has the same structure as the input data.
    """,
        "detect_correct_short": """
    Detect sensor orientation and apply correction to anatomical frame
    """,
        "detect_correct_para": """
    data
        The raw IMU data in the body frame (or arbitrary sensor frame).
        Expected columns: acc_is, acc_ml, acc_pa, gyr_is, gyr_ml, gyr_pa.
    """,
        "detect_correct_return": """
    Returns
    -------
    self
        The instance of the class with the ``corrected_data_`` attribute set to the reoriented data.
    """,
    },
    doc_summary="Decorator to fill common parts of the docstring for subclasses of :class:`BaseReorientationCorrector`.",
)


@base_reorientation_docfiller
class BaseReorientationCorrector(Algorithm):
    """Base class for sensor reorientation correction algorithms.

    This base class should be used for all reorientation correction algorithms.
    Algorithms should implement the ``detect_correct`` method, which detects the sensor
    orientation and applies corrections to align data to the anatomical frame:
        IS → vertical (infero-superior), pointing up
        ML → mediolateral, pointing right
        AP → anteroposterior, pointing forward

    The method should return the instance with the ``corrected_data_`` attribute set.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(corrected_data_)s

    Notes
    -----
    You can use the :func:`~base_reorientation_docfiller` decorator to fill common parts
    of the docstring for your subclass. See the source of this class for an example.

    """

    _action_methods = ("detect_correct",)

    # Other Parameters
    data: pd.DataFrame

    # Results
    corrected_data_: pd.DataFrame

    @base_reorientation_docfiller
    def detect_correct(self, data: pd.DataFrame, **kwargs: Unpack[dict[str, Any]]) -> Self:
        """%(detect_correct_short)s.

        Parameters
        ----------
        %(detect_correct_para)s

        %(detect_correct_return)s
        """
        raise NotImplementedError


__all__ = ["BaseReorientationCorrector", "base_reorientation_docfiller"]
