from types import MappingProxyType
from typing import Any, Final, Optional

import pandas as pd
from tpcp import Pipeline, cf
from tpcp.misc import set_defaults
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import MeasureTimeResults, timed_action_method
from mobgap.signal_based._sdmo import (
    RMS,
    FrequencyAmplitudeWidthSlope,
    HarmonicRatio,
    Jerk,
    RegularitySymmetry,
    SampleEntropy,
    SDRange,
    StrideLevelSDMO,
    TurnSDMO,
)
from mobgap.signal_based.base import BaseSDMOCalculator, base_sdmo_docfiller


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
    %(perf_)s

    """

    _action_methods = ("calculate",)
    _composite_params = ("calculators",)

    calculators: tuple[tuple[str, BaseSDMOCalculator]]
    signal_based_parameters_: pd.DataFrame
    perf_: MeasureTimeResults

    class PredefinedParameters:
        default: Final = MappingProxyType(
            {
                "calculators": (
                    ("sample_entropy", SampleEntropy(dim=2, r=0.15, acc_columns=["acc_is"])),
                    ("harmonic_ratio", HarmonicRatio(acc_columns=["acc_is", "acc_pa"])),
                    ("sd_range", SDRange()),
                    (
                        "jerk",
                        Jerk(
                            acc_columns=["acc_is", "acc_ml", "acc_pa"],
                            gyr_columns=["gyr_is", "gyr_ml", "gyr_pa"],
                        ),
                    ),
                    (
                        "freq_amp_width_slope",
                        FrequencyAmplitudeWidthSlope(acc_columns=["acc_is", "acc_ml", "acc_pa"]),
                    ),
                    ("regularity_symmetry", RegularitySymmetry()),
                    ("rms", RMS()),
                    (
                        "stride_level",
                        StrideLevelSDMO(stride_list_columns=["stride_length_m", "cadence_spm", "stride_duration_s"]),
                    ),
                    ("turn", TurnSDMO()),
                )
            }
        )

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

        if not self.calculators:
            self.signal_based_parameters_ = pd.DataFrame()
            return self

        params = {
            "stride_list": stride_list,
            "sampling_rate_hz": sampling_rate_hz,
            "replicate_matlab": replicate_matlab,
            "turn_list": turn_list,
            **kwargs,
        }
        results = []
        for _name, calculator in self.calculators:
            calculator_ = calculator.clone().calculate(data, **params)
            results.append(calculator_.signal_based_parameters_)
        self.signal_based_parameters_ = pd.concat(results, axis=1)
        return self
