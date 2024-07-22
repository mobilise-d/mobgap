"""Base classes for all Cadence calculation methods."""

from typing import Any

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc

base_cad_docfiller = make_filldoc(
    {
        "other_parameters": """
    data
        The raw IMU data in the body passed to the ``calculate`` method.
    initial_contacts
        The initial contacts passed to the ``calculate`` method.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz passed to the ``calculate`` method.
    """,
        "cadence_per_sec_": """
    cadence_per_sec_
        The main output of the cadence calculation.
        It provides a DataFrame with the column ``cadence_spm`` that contains the cadence values with one value per full
        second of the provided data. The unit is ``1/min``.
        The index of this dataframe is named ``sec_center_samples`` and contains the sample number of the center of the
        each second.
    """,
        "calculate_short": """
    Calculate cadence from the passed data and initial contacts.
    """,
        "calculate_para": """
    data
        The raw IMU data in the body frame.
        We usually assume that this is one gait sequence (i.e. that there are no non-walking periods in the data).
    initial_contacts
        The initial contacts of the gait sequence.
        This should be passed as a DataFrame with the colum ``ic`` that contains the sample number of the initial
        contacts.
        We usually assume that the first IC marks the start of the passed gait sequence and the last IC marks the end.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz.
    """,
        "calculate_return": """
    Returns
    -------
    self
        The instance of the class with the ``cadence_per_sec_`` attribute set to the calculated cadence.
    """,
    },
    doc_summary="Decorator to fill common parts of the docstring for subclasses of :class:`BaseCadenceCalculator`.",
)


@base_cad_docfiller
class BaseCadCalculator(Algorithm):
    """Base class for cadence calculation algorithms.

    This base class should be used for all cadence calculation algorithms.
    Algorithms should implement the ``calculate`` method, which will perform all relevant processing steps.
    The method should then return the instance of the class, with the ``cadence_per_sec_`` attribute set to the
    calculated cadence.
    Further, the calculate method should set ``self.data``, ``self.sampling_rate_hz`` and ``self.initial_contacts``
    to the parameters passed to the method.

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
    %(cadence_per_sec_)s

    Notes
    -----
    You can use the :func:`~base_cad_docfiller` decorator to fill common parts of the docstring for your subclass.
    See the source of this class for an example.

    """

    data: pd.DataFrame
    sampling_rate_hz: float
    initial_contacts: pd.DataFrame

    cadence_per_sec_: pd.DataFrame

    _action_methods = ("calculate",)

    def calculate(
        self,
        data: pd.DataFrame,
        *,
        initial_contacts: pd.DataFrame,
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


__all__ = ["BaseCadCalculator", "base_cad_docfiller"]
