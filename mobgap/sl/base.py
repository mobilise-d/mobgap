"""Base class for step length calculators."""

from typing import Any

import pandas as pd
import numpy as np
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc

base_sl_docfiller = make_filldoc(
    {
        "other_parameters": """
    data
        The raw IMU data of the gait sequence passed to the ``calculate`` method.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz passed to the ``calculate`` method.
    """,
        "sl_sec_list_": """
    sl_sec_list_
        The main output of the step length calculation.
        It provides a DataFrame with the column ``length_m`` that contains the length values with one value per full
        second of the provided data. The unit is ``m``.
        The index of this dataframe is named ``sec_center_samples`` and contains the sample number of the center of the
        each second.    """,
        "sl_list_": """
    sl_list_
        The secondary output of the step length calculation.
        It provides a Numpy array that contains the raw step length values. The unit is ``m``.
        """,
        "calculate_short": """
    Calculate per-sec length values in the passed data.
    """,
        "calculate_para": """
    data
        The raw IMU data of a single sensor.
    initial_contacts
        The indices of the detected initial contacts in the input data.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz.
        """,
        "calculate_return": """
    Returns
    -------
    self
        The instance of the class with the ``sl_sec_list_`` attribute set to the estimated length per second values.
    """,
    },
    doc_summary="Decorator to fill common parts of the docstring for subclasses of :class:`BaseSlCalculator`.",
)


@base_sl_docfiller
class BaseSlCalculator(Algorithm):
    """Base class for SL-calculators.

    This base class should be used for all step length estimation algorithms.
    Algorithms should implement the ``calculate`` method, which will perform all relevant processing steps.
    The method should then return the instance of the class, `sl_sec_list_`` attribute set to the estimated length per
    second values
    Further, the calculate method should set ``self.data`` and ``self.sampling_rate_hz`` to the parameters passed to the
    method.
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
    %(sl_sec_list_)s

    Notes
    -----
    You can use the :func:`~base_sl_docfiller` decorator to fill common parts of the docstring for your subclass.
    See the source of this class for an example.
    """

    _action_methods = ("calculate",)

    # Other Parameters
    data: pd.DataFrame
    initial_contacts: pd.DataFrame
    sampling_rate_hz: float

    # results
    sl_sec_list_: pd.DataFrame
    sl_list_: np.ndarray

    @base_sl_docfiller
    def calculate(self, data: pd.DataFrame, *, initial_contacts: pd.DataFrame, sampling_rate_hz: float, **kwargs: Unpack[dict[str, Any]]) -> Self:
        """%(calculate_short)s.

        Parameters
        ----------
        %(calculate_para)s
        %(calculate_return)s
        """
        raise NotImplementedError


__all__ = ["BaseSlCalculator", "base_sl_docfiller"]
