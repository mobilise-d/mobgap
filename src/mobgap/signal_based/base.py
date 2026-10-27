"""Base class for signal-based digital mobility outcome calculations."""

from typing import Any, ClassVar, Literal, Optional

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc
from mobgap._utils_internal.misc import MeasureTimeResults, timer_doc_filler

base_sdmo_docfiller = make_filldoc(
    {
        "signal_based_parameters_": """
    signal_based_parameters_
        The main output of the signal-based digital mobility outcomes (SDMO) block as a DataFrame with a single row
        and multiple columns containing the implemented signal-based parameters.
        This is a single value per metric per data.
    """,
        "data_param": """
    data
        The raw IMU data passed to the ``calculate`` method.
    """,
        "sampling_rate_param": """
    sampling_rate_hz
        The sampling rate of the IMU data in Hz passed to the ``calculate`` method.
    """,
        "stride_list_param": """
    stride_list
        The stride list associated with the ``data`` passed to the ``calculate`` method.
    """,
        "acc_columns_para": """
    acc_columns
        Name of the acceleration signal columns for which parameters will be calculated.
    """,
        "calculate_short": """
    Calculate parameters for the passed data.
    """,
        "calculate_return": """
    Returns
    -------
    self
        The instance of the class with the ``signal_based_parameters_`` attribute set to the estimated parameters.
    """,
    }
    | timer_doc_filler._dict,
    doc_summary="Decorator to fill common parts of the docstring for subclasses of :class:`BaseSDMOCalculator`.",
)


@base_sdmo_docfiller
class BaseSDMOCalculator(Algorithm):
    """Base class for signal-based digital mobility outcome (SDMO) calculators.

    This base class should be used for all SDMO calculation procedures.
    Algorithms should implement the ``calculate`` method.
    The method should return the instance of the class with the ``signal_based_parameters_`` attribute.
    By default ``result_level="wb"`` selects the full accepted WB in the main pipeline. Turn calculators declare
    ``result_level="turn"``, return their indexed metadata/features in ``per_turn_parameters_`` and leave the
    single-row ``signal_based_parameters_`` empty. They run once over all detected turns on the recording signal.
    Turn calculators preserve the supplied turn-list index, row order and metadata, and use distinct feature names.
    Further, the calculate methods should set all inputs of the calculate method to attributes of the same name.

    Subclasses may specify further parameters for their calculate methods because individual signal-based parameters
    require different inputs.

    Other Parameters
    ----------------
    %(data_param)s
    %(sampling_rate_param)s
    %(stride_list_param)s
    turn_list
        The turn list associated with the ``data`` passed to the ``calculate`` method.
    replicate_matlab
        If True, use MATLAB-compatible smoothing, otherwise the direct pandas-based moving average smoothing.

    Attributes
    ----------
    %(signal_based_parameters_)s

    Notes
    -----
    You can use the :func:`~base_sdmo_docfiller` decorator to fill common parts of the docstring for your subclass.
    See the source of this class for an example.
    """

    _action_methods = ("calculate",)

    # Composite pipelines use this to choose the calculation window.
    result_level: ClassVar[Literal["wb", "turn"]] = "wb"

    # Other Parameters
    data: pd.DataFrame

    # results
    signal_based_parameters_: pd.DataFrame

    perf_: MeasureTimeResults

    @base_sdmo_docfiller
    def calculate(
        self,
        data: pd.DataFrame,
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        """%(calculate_short)s.

        Parameters
        ----------
        %(data_param)s

        %(calculate_return)s
        """
        raise NotImplementedError


__all__ = ["BaseSDMOCalculator", "base_sdmo_docfiller"]


def _turn_metadata(turn_list: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Supplied turn metadata, or the canonical empty standalone schema."""
    if turn_list is None or len(turn_list.columns) == 0:
        turn_list = pd.DataFrame(
            {
                "start": pd.Series(dtype="int64"),
                "end": pd.Series(dtype="int64"),
                "duration_s": pd.Series(dtype=float),
                "angle_deg": pd.Series(dtype=float),
                "direction": pd.Series(dtype=str),
            }
        ).rename_axis("turn_id")
    return turn_list


def _empty_turn_parameters(turn_list: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Empty per-turn result, preserving supplied metadata and index schema."""
    result = _turn_metadata(turn_list).iloc[:0].copy()
    for column in ("turn_mean_ang_vel", "turn_peak_ang_vel", "turn_smoothness"):
        result[column] = pd.Series(index=result.index, dtype=float)
    return result
