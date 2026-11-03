"""Named calculator prototypes for post-processing walking bouts."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from mobgap.secondary_outcomes.base import BaseSecondaryOutcomeCalculator
from mobgap.secondary_outcomes.complexity._sample_entropy import SampleEntropy
from mobgap.secondary_outcomes.periodicity._gait_regularity_and_symmetry import GaitRegularityAndSymmetry
from mobgap.secondary_outcomes.periodicity._harmonic_ratio_peak_trough import HarmonicRatioPeakTrough
from mobgap.secondary_outcomes.periodicity._spectral_peak_features import SpectralPeakFeatures
from mobgap.secondary_outcomes.signal_statistics._root_mean_square import RootMeanSquare
from mobgap.secondary_outcomes.signal_statistics._standard_deviation_and_range import StandardDeviationAndRange
from mobgap.secondary_outcomes.smoothness._jerk_rms_ratio import JerkRMSRatio
from mobgap.secondary_outcomes.smoothness._log_dimensionless_jerk import LogDimensionlessJerk
from mobgap.secondary_outcomes.stride._stride_parameter_cv import StrideParameterCV

#: Default WB calculator prototypes, keyed by the algorithm names used in result columns.
#: Copy the mapping to configure a collection, and clone each entry before calculating a prepared window.
DEFAULT_WB_SDMO_CALCULATORS: Final[Mapping[str, BaseSecondaryOutcomeCalculator]] = MappingProxyType(
    {
        "sample_entropy": SampleEntropy(dim=2, r=0.15, acc_columns=["acc_is"]),
        "harmonic_ratio": HarmonicRatioPeakTrough(acc_columns=["acc_is", "acc_pa"]),
        "sd_range": StandardDeviationAndRange(),
        "log_dimensionless_jerk": LogDimensionlessJerk(
            acc_columns=["acc_is", "acc_ml", "acc_pa"], gyr_columns=["gyr_is", "gyr_ml", "gyr_pa"]
        ),
        "rms_jerk_ratio": JerkRMSRatio(),
        "freq_amp_width": SpectralPeakFeatures(acc_columns=["acc_is", "acc_ml", "acc_pa"]),
        "regularity_symmetry": GaitRegularityAndSymmetry(),
        "rms": RootMeanSquare(acc_columns=["acc_is", "acc_ml", "acc_pa"], gyr_columns=["gyr_is", "gyr_ml", "gyr_pa"]),
        "stride_level": StrideParameterCV(
            stride_list_columns=[
                "stride_length_m",
                "stride_duration_s",
                "cadence_spm",
                "walking_speed_mps",
                "speed_mps",
            ]
        ),
    }
)
