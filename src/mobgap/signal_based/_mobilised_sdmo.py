from tpcp import Pipeline
import dataclasses
from typing import Optional
from typing_extensions import Self
import pandas as pd
from mobgap.signal_based._sdmo import  (
SampleEntropy, HarmonicRatio, SDRange, Jerk, FrequencyAmplitudeWidthSlope, RegularitySymmetry, RMS, StrideLevelSDMO, TurnSDMO
)
from mobgap.signal_based.base import BaseSDMOCalculator

@dataclasses.dataclass
class MobilisedSDMO(Pipeline):
    """Signal-based digital mobility outcome (SDMO) calculations on IMU signal (ideally per walking bout).

    This "algorithm" calculates a predefined set of SDMOs for given signal window.
    """
    _composite_params = ("calculators",)

    signal_based_parameters: pd.DataFrame

    DEFAULT_CALCULATORS = [
        ("sample_entropy", SampleEntropy(dim=2, r=0.15, acc_columns=["acc_is"])),
        ("harmonic_ratio", HarmonicRatio(acc_columns=["acc_is", "acc_pa"])),
        ("sd_range", SDRange()),
        ("jerk", Jerk(acc_columns=["acc_is", "acc_ml", "acc_pa"], gyr_columns=["gyr_is", "gyr_ml", "gyr_pa"])),
        ("freq_amp_width_slope", FrequencyAmplitudeWidthSlope(acc_columns=["acc_is", "acc_ml", "acc_pa"])),
        ("regularity_symmetry", RegularitySymmetry()),
        ("rms", RMS()),
        ("stride_level", StrideLevelSDMO(stride_list_columns=["stride_length_m", "cadence_spm", "stride_duration_s"])),
        ("turn", TurnSDMO())
    ]

    def __init__(self, calculators: Optional[list[tuple[str, BaseSDMOCalculator]]] = None):
        self.calculators = calculators

    def calculate(
            self,
            data: pd.DataFrame,
            sampling_rate_hz: float,
            stride_list: Optional[pd.DataFrame] = None,
            turn_list: Optional[pd.DataFrame] = None,
            replicate_matlab: Optional[bool] = True,
            **kwargs) -> Self:
        params = {
            "stride_list": stride_list,
            "sampling_rate_hz": sampling_rate_hz,
            "replicate_matlab": replicate_matlab,
            "turn_list": turn_list,
            **kwargs
        }
        calc_list  = self.DEFAULT_CALCULATORS if self.calculators is None else self.calculators
        results = []
        for name, calculator in calc_list :
            calculator_ = calculator.clone().calculate(data, **params)
            results.append(calculator_.signal_based_parameters)
        self.signal_based_parameters = pd.concat(results, axis=1)
        return self