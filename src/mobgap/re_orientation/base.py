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
    doc_summary="Decorator to fill common parts of the docstring for subclasses of "
    ":class:`BaseReorientationCorrector`.",
)


@base_reorientation_docfiller
class BaseReorientationCorrector(Algorithm):
    """Base class for sensor reorientation correction algorithms.

    This base class should be used for all reorientation correction algorithms.
    Algorithms should implement the ``detect_correct`` method, which detects the sensor
    orientation and applies corrections to align data to the anatomical frame:

    - IS → vertical (infero-superior), pointing up
    - ML → mediolateral, pointing right
    - AP → anteroposterior, pointing forward

    The method should return the instance with the ``corrected_data_`` attribute set.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(corrected_data_)s

    Notes
    -----
    Reorientation is performed after gait sequence detection rather than on the full recording.
    This design choice is optimal because:

    1. Mobilise-D DMOs are calculated within walking bouts, making correction outside them
       unnecessary
    2. The reorientation method requires a known reference posture - upright walking during
       detected gait sequences provides this reference when device mounting orientation is unknown

    GsdIonescu is orientation-independent and reliably detects gait sequences regardless of sensor
    orientation. However, GsdIluz is orientation-dependent and may fail to detect gait sequences in
    non-standard orientations, which is a known limitation of combining these algorithms.

    You can use the :func:`~base_reorientation_docfiller` decorator...
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
