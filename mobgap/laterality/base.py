"""Base class for LR detectors."""

from collections.abc import Iterable
from typing import Any, Union

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc

base_lrc_docfiller = make_filldoc(
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
        The predicted left and right foot initial contacts.
        The dataframe is identical to the input ``ic_list``, but with the ``lr`` column added.
        The ``lr`` column specifies if the respective IC belongs to the left or the right foot.
    """,
        "predict_short": """
        Assign a left/right label to each initial contact in the passed data.
    """,
        "predict_para": """
    data
        The raw IMU of a single sensor.
        This should usually represent a single gait sequence or walking bout.
    ic_list
        The list of initial contacts within the data.
        The ``ic_list`` is expected to have a column ``ic`` with the indices of the detected initial contacts relative
        to the start of the passed data.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz.
    """,
        "predict_return": """
    Returns
    -------
    self
        The instance of the class with the ``ic_lr_list_`` attribute set to the passed ICs with a new left/right column.
    """,
        "self_optimize_paras": """
    data_sequences
        A sequence/iterable/list of dataframes, each containing the raw IMU data of a single sensor.
        Each sequence should usually contain the data of a single gait sequence/walking bout.
        The optimization will be performed over all sequences combined.
    ic_list_per_sequence
        A sequence/iterable/list of gsd-list, each containing the list of detected ics for the respective
        data sequence.
        The ``ic_list`` is expected to have a column ``ic`` with the indices of the detected initial contacts relative
        to the start of the each passed data sequence.
    ref_ic_lr_list_per_sequence
        A sequence/iterable/list of reference ic_lr_list, each containing the reference left/right initial contacts.
        They are expected to have the exact same structure as the ic_lists passed as ``ic_list_per_sequence``, but
        should contain the ground-truth left/right labels in a additional column called ``lr_label``.
        They are used as ground-truth to validate the output of the algorithm during optimization.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz.
        This can either be a single float, in case all sequences have the same sampling rate, or a sequence of
        floats, in case the sampling rate differs between the sequences.
        """,
        "self_optimize_return": """
    Returns
    -------
    self
        The instance of the class with the internal parameters optimized.
    """,
    },
    doc_summary="Decorator to fill common parts of the docstring for subclasses of :class: `BaseLRDetector`.",
)


@base_lrc_docfiller
class BaseLRClassifier(Algorithm):
    """Base class for L/R foot classifier.

    This base class should be used for all Left/Right foot classificaiton algorithms.
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

    _action_methods = ("predict",)

    data: pd.DataFrame
    ic_list: pd.DataFrame
    sampling_rate_hz: float

    # results
    ic_lr_list_: pd.DataFrame

    @base_lrc_docfiller
    def predict(
        self,
        data: pd.DataFrame,
        ic_list: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        """%(predict_short)s.

        Parameters
        ----------
        %(predict_para)s
        %(predict_return)s
        """
        raise NotImplementedError

    @base_lrc_docfiller
    def self_optimize(
        self,
        data_sequences: Iterable[pd.DataFrame],
        ic_list_per_sequence: Iterable[pd.DataFrame],
        ref_ic_lr_list_per_sequence: Iterable[pd.DataFrame],
        *,
        sampling_rate_hz: Union[float, Iterable[float]],
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        """Optimize the internal parameters of the algorithm.

        This is only relevant for algorithms that have a special internal optimization approach (like ML based algos).

        Parameters
        ----------
        %(self_optimize_paras)s

        %(self_optimize_return)s

        """
        raise NotImplementedError("This algorithm does not implement a internal optimization.")


_ic_lr_list_dtypes = {
    "ic": "int64",
    "lr_label": pd.CategoricalDtype(categories=["left", "right"]),
}


def _unify_ic_lr_list_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(_ic_lr_list_dtypes)[list(_ic_lr_list_dtypes.keys())]


__all__ = ["BaseLRClassifier", "base_lrc_docfiller"]
