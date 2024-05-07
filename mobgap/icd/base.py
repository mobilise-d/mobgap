"""Base class for ICs detectors."""

from typing import Any

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc

base_icd_docfiller = make_filldoc(
    {
        "other_parameters": """
    data
        The raw IMU data of the gait sequence passed to the ``detect`` method.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz passed to the ``detect`` method.
    """,
        "ic_list_": """
    ic_list_
        A pandas dataframe with the indices of the detected initial contacts in the input data.
        It only has one column, ``ic``, which contains the indices of the detected initial contacts.
    """,
        "detect_short": """
    Detect Initial contacts in the passed data
    """,
        "detect_info": """
    We expect the data to be a single gait sequence.
    If the data does not contain any gait sequences, the algorithm might behave unexpectedly.
    """,
        "detect_para": """
    data
        The raw IMU of a single sensor.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz.
    """,
        "detect_return": """
    Returns
    -------
    self
        The instance of the class with the ``icd_list_`` attribute set to the detected initial contacts.
    """,
    },
    doc_summary="Decorator to fill common parts of the docstring for subclasses of :class:`BaseIcdDetector`.",
)


@base_icd_docfiller
class BaseIcDetector(Algorithm):
    """Base class for IC-detectors.

    This base class should be used for all initial contacts detection algorithms.
    Algorithms should implement the ``detect`` method, which will perform all relevant processing steps.
    The method should then return the instance of the class, with the ``ic_list_`` attribute set to the detected
    initial contacts.
    Further, the detect method should set ``self.data`` and ``self.sampling_rate_hz`` to the parameters passed to the
    method.
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
    %(ic_list_)s

    Notes
    -----
    You can use the :func:`~base_icd_docfiller` decorator to fill common parts of the docstring for your subclass.
    See the source of this class for an example.
    """

    _action_methods = ("detect",)

    # Other Parameters
    data: pd.DataFrame
    sampling_rate_hz: float

    # results
    ic_list_: pd.DataFrame

    @base_icd_docfiller
    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **kwargs: Unpack[dict[str, Any]]) -> Self:
        """%(detect_short)s.

        Parameters
        ----------
        %(detect_para)s
        %(detect_return)s
        """
        raise NotImplementedError

    @property
    def sub_gs_list_(self):
        ics = self.ic_list_["ic"]
        return pd.DataFrame.from_records([{"gs_id": "1", "start": ics.iloc[0], "end": ics.iloc[-1] + 1}]).set_index(
            "gs_id"
        )

    @property
    def ic_list_per_sub_gs_(self):
        # Note: We provide an overkill implementation for the default case (just a single sub gs).
        #       However, this way, you don't need to reimplement this method, when a more complex sub gs logic is
        #       required.
        # For each sub-gs, we get all ICs that are within the sub-gs.
        gs_list = self.sub_gs_list_.copy().reset_index()
        gs_list.index = pd.IntervalIndex.from_arrays(gs_list["start"], gs_list["end"], closed="left")

        def ics_relative_to_gs(df):
            matches = gs_list.loc[df["ic"]]
            return df.assign(gs_id=matches["gs_id"].to_numpy(), ic=df["ic"] - matches["start"].to_numpy())

        ics_with_gs_label = self.ic_list_.pipe(ics_relative_to_gs)
        return ics_with_gs_label.set_index("gs_id", append=True).reorder_levels(["gs_id", "step_id"])


__all__ = ["BaseIcDetector", "base_icd_docfiller"]
