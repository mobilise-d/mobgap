from types import MappingProxyType
from typing import Any, Final, Optional

import pandas as pd
from tpcp import Pipeline, cf
from tpcp.misc import set_defaults
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import MeasureTimeResults, timed_action_method
from mobgap.signal_based._presets import DEFAULT_WB_SDMO_CALCULATORS
from mobgap.signal_based._sdmo import TurnSDMO
from mobgap.signal_based.base import BaseSDMOCalculator, _empty_turn_parameters, _turn_metadata, base_sdmo_docfiller


@base_sdmo_docfiller
class MobilisedSDMO(Pipeline):
    """Signal-based digital mobility outcome (SDMO) calculations on IMU signal (ideally per walking bout).

    This "algorithm" calculates a predefined set of SDMOs for given signal window.

    Parameters
    ----------
    calculators
        List of available calculators that are subclasses of :class:`BaseSDMOCalculator`.

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
    per_turn_parameters_
        Separate rows for supplied turns, preserving their index, metadata and coordinates relative to ``data``.
        Only calculators with ``result_level="turn"`` contribute; no per-turn values enter the bout result.
        Empty when no turn calculator is configured or no turns are supplied.
    %(perf_)s

    """

    _action_methods = ("calculate",)
    _composite_params = ("calculators",)

    calculators: tuple[tuple[str, BaseSDMOCalculator]]
    signal_based_parameters_: pd.Series
    per_turn_parameters_: pd.DataFrame
    perf_: MeasureTimeResults

    class PredefinedParameters:
        default: Final = MappingProxyType({"calculators": (*DEFAULT_WB_SDMO_CALCULATORS.items(), ("turn", TurnSDMO()))})

    @set_defaults(calculators=cf(PredefinedParameters.default["calculators"]))
    def __init__(self, calculators: tuple[tuple[str, BaseSDMOCalculator]]) -> None:
        self.calculators = calculators

    @base_sdmo_docfiller
    @timed_action_method
    def calculate(
        self,
        data: pd.DataFrame,
        sampling_rate_hz: float,
        stride_list: Optional[pd.DataFrame] = None,
        turn_list: Optional[pd.DataFrame] = None,
        replicate_matlab: Optional[bool] = True,
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.stride_list = stride_list
        self.turn_list = turn_list
        self.replicate_matlab = replicate_matlab

        self.per_turn_parameters_ = _empty_turn_parameters(turn_list)
        if not self.calculators:
            self.signal_based_parameters_ = pd.Series(dtype=float)
            return self

        params = {
            "stride_list": stride_list,
            "sampling_rate_hz": sampling_rate_hz,
            "replicate_matlab": replicate_matlab,
            "turn_list": turn_list,
            **kwargs,
        }
        results = [
            calculator.clone().calculate(data, **params).signal_based_parameters_
            for _name, calculator in self.calculators
            if calculator.result_level != "turn"
        ]
        self.signal_based_parameters_ = pd.concat(results) if results else pd.Series(dtype=float)
        self.per_turn_parameters_ = _calculate_turn_parameters(
            data,
            turn_list,
            tuple((name, calc) for name, calc in self.calculators if calc.result_level == "turn"),
            **{key: value for key, value in params.items() if key != "turn_list"},
        )
        return self


def _calculate_turn_parameters(
    data: pd.DataFrame,
    turn_list: Optional[pd.DataFrame],
    calculators: tuple[tuple[str, BaseSDMOCalculator], ...],
    **params: Any,
) -> pd.DataFrame:
    """Assemble single-window turn calculators for the optional bulk composition interface."""
    metadata = _turn_metadata(turn_list)
    if metadata.empty or not calculators:
        return _empty_turn_parameters(turn_list)
    rows_per_calculator = [[] for _ in calculators]
    for start, end in metadata[["start", "end"]].itertuples(index=False, name=None):
        window = data.iloc[int(start) : int(end)]
        for rows, (_name, calculator) in zip(rows_per_calculator, calculators, strict=True):
            rows.append(calculator.clone().calculate(window, **params).signal_based_parameters_)
    # Assemble each calculator separately so floating-point metrics do not coerce integer feature columns.
    frames = [pd.DataFrame(rows, index=metadata.index) for rows in rows_per_calculator]
    return pd.concat([metadata, *frames], axis=1)
