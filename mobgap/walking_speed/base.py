"""Base class for walking speed calculators."""

from typing import Any, Optional

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc

base_ws_docfiller = make_filldoc(
    {
        "walking_speed_per_sec_": """
    walking_speed_per_sec_
        The main output of the walking speed calculation.
        It provides a DataFrame with the column ``walking_speed_mps`` that contains the walking speed values with one
        value per full second of the provided data. The unit is ``m/s``.
        The index of this dataframe is named ``sec_center_samples`` and contains the sample number of the center of the
        each second.
    """,
        "other_parameters": """
    data
        The raw IMU data of the gait sequence passed to the ``calculate`` method.
    initial_contacts
        The initial contacts passed to the ``calculate`` method.
    cadence_per_sec
        The cadence values provided to the ``calculate`` method.
    stride_length_per_sec
        The stride length values provided to the ``calculate`` method.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz passed to the ``calculate`` method.
    """,
        "calculate_short": """
    Calculate per-sec walking speed values in the passed data.
    """,
        "calculate_para": """
    data
        The raw IMU data of a single sensor.
        We usually assume that this is one gait sequence (i.e. that there are no non-walking periods in the data).
    initial_contacts
        The initial contacts of the gait sequence.
        This should be passed as a DataFrame with the colum ``ic`` that contains the sample number of the initial
        contacts.
        We usually assume that the first IC marks the start of the passed gait sequence and the last IC marks the end.
    cadence_per_sec
        The cadence per sec within the gait sequence.
        This should be a DataFrame with a ``cadence_spm`` column containing one cadence value per second in the GS.
    stride_length_per_sec
        This should be a DataFrame with a ``stride_length_m`` column containing one stride length value per second
        in the GS.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz.
    """,
        "calculate_return": """
    Returns
    -------
    self
        The instance of the class with the ``walking_speed_per_sec_`` attribute set to the estimated walking speed
        per second values.
    """,
    }
)


@base_ws_docfiller
class BaseWsCalculator(Algorithm):
    """Base class for Walking Speed calculators.

    This base class should be used for all walking speed estimation algorithms.
    Algorithms should implement the ``calculate`` method.
    The method should return the instance of the class with the ``walking_speed_per_sec_`` attribute set to the walking
    speed values per second.
    Further, the calculate methods should set all inputs of the calculate method to attributes of the same name.

    We allow that subclasses specify further parameters for the calculate methods (hence, this baseclass supports
    ``**kwargs``).
    However, you should only use them, if you really need them and apply active checks, that they are passed correctly.
    In 99%% of the time, you should add a new parameter to the algorithm itself, instead of adding a new parameter to
    the calculate method.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(walking_speed_per_sec_)s

    Notes
    -----
    You can use the :func:`~base_ws_docfiller` decorator to fill common parts of the docstring for your subclass.
    See the source of this class for an example.
    """

    _action_methods = ("calculate",)

    data: pd.DataFrame
    initial_contacts: pd.DataFrame
    sampling_rate_hz: float
    cadence_per_sec: Optional[pd.DataFrame]
    stride_length_per_sec: Optional[pd.DataFrame]

    # results
    walking_speed_per_sec_: pd.DataFrame

    @base_ws_docfiller
    def calculate(
        self,
        data: pd.DataFrame,
        *,
        initial_contacts: Optional[pd.DataFrame] = None,
        cadence_per_sec: Optional[pd.DataFrame] = None,
        stride_length_per_sec: Optional[pd.DataFrame] = None,
        sampling_rate_hz: float,
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        """%(calculate_short)s.

        Parameters
        ----------
        %(calculate_para)s
        %(calculate_return)s
        """
        raise NotImplementedError


__all__ = ["BaseWsCalculator", "base_ws_docfiller"]
