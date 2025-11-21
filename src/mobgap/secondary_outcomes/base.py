"""Base class for walking speed calculators."""

from typing import Any, Optional

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc

base_sdmo_docfiller = make_filldoc(
    {
        "secondary_outcomes": """
    secondary_outcomes
        The main output of the secondary outcomes block.
        It provides a DataFrame with the columns containing the implemented secondary digital mobility outcomes
        per provided data (ideally the walking bout, but can work with any data). Units are defined for each outcome.
    """,
        "other_parameters": """
    data
        The raw IMU data of ideally the walking bout passed to the ``calculate`` method.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz passed to the ``calculate`` method.
    """,
        "calculate_short": """
    Calculate secondary outcomes for the passed data.
    """,
        "calculate_para": """
    data
        The raw IMU data of a single sensor.
        We usually assume that this is one final walking bout after running the main blocks.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz.
    """,
        "calculate_return": """
    Returns
    -------
    self
        The instance of the class with the ``secondary_outcomes`` attribute set to the estimated secondary outcomes.
    """,
    }
)


@base_sdmo_docfiller
class BaseSDMOCalculator(Algorithm):
    """Base class for secondary DMO calculators.

    This base class should be used for all secondary outcome calculation procedures/classes (currently one because all
    outcomes will be calculated together as there is no need for dividing them into groups).
    Algorithms should implement the ``calculate`` method.
    The method should return the instance of the class with the ``secondary_outcomes`` attribute set to the walking
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
    %(secondary_outcomes)s

    Notes
    -----
    You can use the :func:`~base_sdmo_docfiller` decorator to fill common parts of the docstring for your subclass.
    See the source of this class for an example.
    """

    _action_methods = ("calculate",)

    data: pd.DataFrame
    sampling_rate_hz: float

    # results
    secondary_outcomes: pd.DataFrame

    @base_sdmo_docfiller
    def calculate(
        self,
        data: pd.DataFrame,
        *,
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


__all__ = ["BaseSDMOCalculator", "base_sdmo_docfiller"]
