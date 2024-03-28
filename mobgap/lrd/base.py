"""Base class for LR detectors."""

from typing import Any

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc

base_lrd_docfiller = make_filldoc(
    {
        "other_parameters": """
    data
        The raw IMU data passed to the ``detect`` method.
    ic_list
        The list of initial contacts passed to the ``detect`` method.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz passed to the ``detect`` method.
    """,
        "ic_lr_list_": """
    ic_lr_list_
        A dataframe specifying the detected left and right foot initial contacts.
        The dataframe is identical to the input ``ic_list``, but with the ``lr`` column added.
        The ``lr`` column specifies if the respective IC belongs to the left or the right foot.
    """,
        "detect_short": """
        Assign a left/right label to each initial contact in the passed data.
    """,
        "detect_para": """
    data
        The raw IMU of a single sensor.
    ic_list
        The list of initial contacts within the data.
        The ``ic_list`` is expected to have a column ``ic`` with the indices of the detected initial contacts relative
        to the start of the passed data.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz.
    """,
        "detect_return": """
    Returns
    -------
    self
        The instance of the class with the ``ic_lr_list_`` attribute set to the detected left/right initial contacts.
    """,
    },
    doc_summary="Decorator to fill common parts of the docstring for subclasses of :class: `BaseLRDetector`.",
)


@base_lrd_docfiller
class BaseLRDetector(Algorithm):
    """Base class for L/R foot detectors.

    This base class should be used for all Left/Right foot detection algorithms.
    Algorithms should implement the ``detect`` method, which will perform all relevant processing steps.
    The method should then return the instance of the class, with the ``ic_lr_list_`` attribute set to the ``ic_list``
    provided by the used with a new addition `lr` column that either contains the string ``left`` or ``right``,
    indicating the laterality of the initial contact.

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
    %(ic_lr_list_)s

    Notes
    -----
    You can use the :func:`~base_lrd_docfiller` decorator to fill common parts of the docstring for your subclass.
    See the source of this class for an example.

    """

    _action_methods = ("detect",)

    data: pd.DataFrame
    ic_list: pd.DataFrame
    sampling_rate_hz: float

    # results
    ic_lr_list_: pd.DataFrame

    @base_lrd_docfiller
    def detect(
        self,
        data: pd.DataFrame,
        ic_list: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        """%(detect_short)s.

        Parameters
        ----------
        %(detect_para)s
        %(detect_return)s
        """
        raise NotImplementedError


__all__ = ["BaseLRDetector", "base_lrd_docfiller"]
