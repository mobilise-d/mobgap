"""Base class for GSD detectors."""

from collections.abc import Iterable
from typing import Any, Union

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc
from mobgap._utils_internal.misc import MeasureTimeResults, timer_doc_filler

base_gsd_docfiller = make_filldoc(
    {
        "other_parameters": """
    data
        The raw IMU data in the body frame passed to the ``detect`` method.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz passed to the ``detect`` method.
    """,
        "gs_list_": """
    gs_list_
        A dataframe specifying the detected gait sequences.
        The dataframe has a ``start`` and ``end`` column, specifying the start and end index of the gait sequence.
        The values are specified as samples after the start of the recording (i.e. the start of the ``data``).
    """,
        "detect_short": """
    Detect gait sequences in the passed data
    """,
        "detect_para": """
    data
        The raw IMU data in the body frame.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz.
    """,
        "detect_return": """
    Returns
    -------
    self
        The instance of the class with the ``gs_list_`` attribute set to the detected gait sequences.
    """,
        "self_optimize_paras": """
    data_sequences
        A sequence/iterable/list of dataframes, each containing the raw IMU data of a single sensor.
        This could be individual trials or data from different participants.
        The optimization will be performed over all sequences combined.
    ref_gsd_list_per_sequence
        A sequence/iterable/list of gsd-list, each containing the reference gait sequences for the respective
        data sequence.
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
    }
    | timer_doc_filler._dict,
    doc_summary="Decorator to fill common parts of the docstring for subclasses of :class:`BaseGsdDetector`.",
)


@base_gsd_docfiller
class BaseGsDetector(Algorithm):
    """Base class for GS-detectors.

    This base class should be used for all gait sequence detection algorithms.
    Algorithms should implement the ``detect`` method, which will perform all relevant processing steps.
    The method should then return the instance of the class, with the ``gs_list_`` attribute set to the detected
    gait sequences.
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
    %(gs_list_)s

    Notes
    -----
    You can use the :func:`~base_gsd_docfiller` decorator to fill common parts of the docstring for your subclass.
    See the source of this class for an example.

    """

    _action_methods = ("detect",)

    # Other Parameters
    data: pd.DataFrame
    sampling_rate_hz: float

    # results
    gs_list_: pd.DataFrame

    perf_: MeasureTimeResults

    @base_gsd_docfiller
    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **kwargs: Unpack[dict[str, Any]]) -> Self:
        """%(detect_short)s.

        Parameters
        ----------
        %(detect_para)s

        %(detect_return)s
        """
        raise NotImplementedError

    @base_gsd_docfiller
    def self_optimize(
        self,
        data_sequences: Iterable[pd.DataFrame],
        ref_gsd_list_per_sequence: Iterable[pd.DataFrame],
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


def get_gs_df_dtypes(expected_id_name: str) -> dict[str, str]:
    return {
        expected_id_name: "int64",
        "start": "int64",
        "end": "int64",
    }


def _unify_gs_df(df: pd.DataFrame, expected_id_name: str = "gs_id") -> pd.DataFrame:
    if expected_id_name not in df.columns and expected_id_name not in df.index.names:
        df = df.rename_axis(expected_id_name).reset_index()
    elif expected_id_name not in df.columns:
        df = df.reset_index()
    gs_df_dtypes = get_gs_df_dtypes(expected_id_name)
    return df.astype(gs_df_dtypes)[list(gs_df_dtypes.keys())].set_index(expected_id_name)


__all__ = ["BaseGsDetector", "base_gsd_docfiller"]
