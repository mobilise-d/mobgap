"""Base class for weartime detectors."""

import warnings
from collections.abc import Iterable
from typing import Any, Union

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc
from mobgap._utils_internal.misc import MeasureTimeResults, timer_doc_filler
from mobgap.weartime.utils import clip_intervals_to_waking_hours

base_weartime_docfiller = make_filldoc(
    {
        "other_parameters": """
data
    The raw IMU data in the body frame passed to the ``detect`` method.
sampling_rate_hz
    The sampling rate of the IMU data in Hz passed to the ``detect`` method.
""",
        "weartime_list_": """
weartime_list_
    A dataframe specifying the detected weartime periods.
    The dataframe has an index ``wt_id`` and columns ``start`` and ``end``, specifying the start and end
    index of each weartime period.
    The values are specified as samples after the start of the recording (i.e. the start of the ``data``).
""",
        "total_weartime_samples_": """
total_weartime_samples_
    The total weartime in samples across all detected weartime periods.
""",
        "total_weartime_min_": """
total_weartime_min_
    The total weartime in minutes across all detected weartime periods.
""",
        "total_weartime_during_waking_min_": """
total_weartime_during_waking_min_
    Total wear-time during the configured waking-hours window in minutes.
    For recordings that do not cover the full configured window, this can fall back to ``total_weartime_min_``.
    For recordings longer than one day, algorithms should raise an error instead of applying a single daily window.
""",
        "detect_short": """
Detect weartime periods in the passed data
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
    The instance of the class with the ``weartime_list_``, ``total_weartime_samples_``,
    ``total_weartime_min_``, and ``total_weartime_during_waking_min_`` properties available for the detected
    weartime periods and total weartime values.
""",
        "self_optimize_paras": """
data_sequences
    A sequence/iterable/list of dataframes, each containing the raw IMU data of a single sensor.
    This could be individual trials or data from different participants.
    The optimization will be performed over all sequences combined.
ref_weartime_list_per_sequence
    A sequence/iterable/list of weartime-lists, each containing the reference weartime periods for the respective
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
        **timer_doc_filler._dict,
    },
    doc_summary="Decorator to fill common parts of the docstring for subclasses of :class:`BaseWeartimeDetector`.",
)


@base_weartime_docfiller
class BaseWeartimeDetector(Algorithm):
    """Base class for weartime detectors.

    This base class should be used for all weartime detection algorithms.
    Algorithms should implement the ``detect`` method, which will perform all relevant processing steps.
    The method should then return the instance of the class with ``weartime_list_`` set to the detected weartime
    periods. Summary statistics are exposed as properties derived from ``weartime_list_`` and ``sampling_rate_hz``.

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
    %(weartime_list_)s
    %(total_weartime_samples_)s
    %(total_weartime_min_)s
    %(total_weartime_during_waking_min_)s
    %(perf_)s

    Notes
    -----
    **Waking Hours Calculation**

    All algorithms calculate wear-time during waking hours in addition to total wear-time.
    This is required for Mobilise-D Digital Mobility Assessment (DMA) validation, which requires ≥12 hours of wear-time
    during waking hours per valid day. Algorithms may expose a configurable waking-hours window; the current default is
    07:00-22:00.

    The waking hours calculation assumes recordings are segmented per day (midnight-to-midnight).
    For recordings shorter than the configured waking-hours end, algorithms issue a warning and use
    ``total_weartime_min_`` as a fallback for ``total_weartime_during_waking_min_``.
    For recordings longer than one day, algorithms should raise an error instead of applying a single daily window to
    multi-day data.

    **Implementation Notes**

    You can use the :func:`~base_weartime_docfiller` decorator to fill common parts of the docstring for your subclass.
    See the source of this class for an example.

    """

    _action_methods = ("detect",)

    # Other Parameters
    data: pd.DataFrame
    sampling_rate_hz: float

    # Results
    weartime_list_: pd.DataFrame
    total_weartime_samples_: int
    total_weartime_min_: float
    total_weartime_during_waking_min_: float

    perf_: MeasureTimeResults

    @property
    def total_weartime_samples_(self) -> int:
        """The total weartime in samples across all detected weartime periods."""
        return int((self.weartime_list_["end"] - self.weartime_list_["start"]).sum())

    @property
    def total_weartime_min_(self) -> float:
        """The total weartime in minutes across all detected weartime periods."""
        return self.total_weartime_samples_ / (60 * self.sampling_rate_hz)

    @property
    def total_weartime_during_waking_min_(self) -> float:
        """Wear-time in minutes during the configured daily waking-hours window."""
        try:
            data = self.data
            sampling_rate_hz = self.sampling_rate_hz
            waking_hours_min = self.waking_hours_min
        except AttributeError as exc:
            raise AttributeError(
                "`total_weartime_during_waking_min_` is only available after calling `detect` on an algorithm with "
                "`data`, `sampling_rate_hz`, and `waking_hours_min` available."
            ) from exc

        data_length = len(data)
        recording_hours = data_length / (3600 * sampling_rate_hz)
        if recording_hours > 24:
            raise ValueError(
                "Cannot calculate weartime during waking hours for recordings longer than one day. "
                "Segment the recording into individual days before applying a daily waking-hours window."
            )

        waking_end_sample = int(waking_hours_min[1] * 60 * sampling_rate_hz)
        if not isinstance(data.index, pd.DatetimeIndex) and data_length < waking_end_sample:
            warnings.warn(
                f"Recording duration ({recording_hours:.1f}h) is shorter than the configured waking-hours window. "
                "Using total_weartime_min_ for total_weartime_during_waking_min_.",
                stacklevel=2,
            )
            return self.total_weartime_min_

        weartime_waking = clip_intervals_to_waking_hours(
            self.weartime_list_,
            data=data,
            sampling_rate_hz=sampling_rate_hz,
            waking_hours_min=waking_hours_min,
        )
        total_weartime_waking_samples = (weartime_waking["end"] - weartime_waking["start"]).sum()
        return total_weartime_waking_samples / (60 * sampling_rate_hz)

    @base_weartime_docfiller
    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **kwargs: Unpack[dict[str, Any]]) -> Self:
        """%(detect_short)s.

        Parameters
        ----------
        %(detect_para)s

        %(detect_return)s
        """
        raise NotImplementedError

    @base_weartime_docfiller
    def self_optimize(
        self,
        data_sequences: Iterable[pd.DataFrame],
        ref_weartime_list_per_sequence: Iterable[pd.DataFrame],
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


def get_weartime_df_dtypes(expected_id_name: str = "wt_id") -> dict[str, str]:
    """Get the expected data types for a weartime dataframe.

    Parameters
    ----------
    expected_id_name
        The name of the ID column for weartime periods.

    Returns
    -------
    dict[str, str]
        A dictionary mapping column names to their expected data types.
    """
    return {
        expected_id_name: "int64",
        "start": "int64",
        "end": "int64",
    }


def _unify_weartime_df(df: pd.DataFrame, expected_id_name: str = "wt_id") -> pd.DataFrame:
    """Unify the format of a weartime dataframe.

    This function ensures that the weartime dataframe has the expected format with proper
    column names, data types, and index.

    Parameters
    ----------
    df
        The weartime dataframe to unify.
    expected_id_name
        The expected name for the weartime ID column.

    Returns
    -------
    pd.DataFrame
        The unified weartime dataframe with the ID as index.
    """
    if expected_id_name not in df.columns and expected_id_name not in df.index.names:
        df = df.rename_axis(expected_id_name).reset_index()
    elif expected_id_name not in df.columns:
        df = df.reset_index()
    weartime_df_dtypes = get_weartime_df_dtypes(expected_id_name)
    return df.astype(weartime_df_dtypes)[list(weartime_df_dtypes.keys())].set_index(expected_id_name)


__all__ = ["BaseWeartimeDetector", "_unify_weartime_df", "base_weartime_docfiller", "get_weartime_df_dtypes"]
