"""Base class for reorientation correction algorithms."""

from typing import Any, Generic, Self, TypeVar, Unpack

import pandas as pd
from tpcp import Algorithm

from mobgap._docutils import make_filldoc

base_reorientation_docfiller = make_filldoc(
    {
        "other_parameters": """
data
    The raw IMU data in the sensor frame passed to the ``detect_correct`` method.
sampling_rate_hz
    The sampling rate of the raw IMU data in Hz.
""",
        "corrected_data_": """
corrected_data_
    The reoriented IMU data in the body frame.
    The dataframe has the same index as the input data and body-frame columns.
""",
        "correction_outputs": """
correction_
    The method-specific value that replays this run's correction through ``apply_correction``.
    It must not retain the input or corrected signal data. Fallback corrections are included.
correction_applied_
    Whether this run applied a non-nominal orientation correction. False still permits the nominal
    sensor-to-body-frame conversion. This flag does not indicate whether orientation was resolved.
""",
        "detect_correct_short": """
Detect sensor orientation and apply correction to the body frame
""",
        "detect_correct_para": """
data
    The raw IMU data in the sensor frame.
    Expected columns: acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z.
sampling_rate_hz
    The sampling rate of the raw IMU data in Hz.
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


CorrectionT = TypeVar("CorrectionT")


@base_reorientation_docfiller
class BaseReorientationCorrector(Algorithm, Generic[CorrectionT]):
    """Base class for sensor reorientation correction algorithms.

    This base class should be used for all reorientation correction algorithms.
    Algorithms should implement the ``detect_correct`` method, which detects the sensor
    orientation and applies corrections to align sensor-frame data to the body frame:

    - x → vertical (infero-superior), pointing up
    - y → mediolateral, pointing right
    - z → posterior-anterior, pointing forward

    The method should return the instance with the ``corrected_data_`` attribute set to body-frame data.
    Each method defines its own correction value type through ``CorrectionT`` and must
    implement the static ``apply_correction(data, correction)`` operation. This operation
    replays the method's correction on sensor-frame data without estimating orientation
    or needing an executed algorithm instance. Correction values can describe fallback
    behavior, so applying one does not prove that orientation was resolved.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(corrected_data_)s
    %(correction_outputs)s

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
    sampling_rate_hz: float

    # Results
    corrected_data_: pd.DataFrame
    correction_: CorrectionT
    correction_applied_: bool

    @staticmethod
    def apply_correction(data: pd.DataFrame, correction: CorrectionT) -> pd.DataFrame:
        """Apply a method-specific correction without estimating orientation.

        Parameters
        ----------
        data
            Sensor-frame IMU data with the columns supported by the concrete method.
        correction
            A correction value provided by this method, in its own representation.
            Other methods may define different correction types.

        Returns
        -------
        pandas.DataFrame
            Corrected body-frame data with the same index as ``data``. The input
            must not be modified. Applying a fallback correction does not imply
            that orientation was resolved.
        """
        raise NotImplementedError

    @base_reorientation_docfiller
    def detect_correct(self, data: pd.DataFrame, *, sampling_rate_hz: float, **kwargs: Unpack[dict[str, Any]]) -> Self:
        """%(detect_correct_short)s.

        Parameters
        ----------
        %(detect_correct_para)s

        %(detect_correct_return)s
        """
        raise NotImplementedError


__all__ = ["BaseReorientationCorrector", "base_reorientation_docfiller"]
