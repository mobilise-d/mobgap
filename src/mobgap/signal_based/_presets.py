"""Named calculator prototypes for post-processing walking bouts."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from mobgap.signal_based._sdmo import (
    RMS,
    FrequencyAmplitudeWidth,
    HarmonicRatio,
    LogDimensionlessJerk,
    RegularitySymmetry,
    RMSJerkRatio,
    SampleEntropy,
    SDRange,
    StrideLevelSDMO,
)
from mobgap.signal_based.base import BaseSDMOCalculator

#: Default WB calculator prototypes, keyed by the algorithm names used in result columns.
#: Copy the mapping to configure a collection, and clone each entry before calculating a prepared window.
DEFAULT_WB_SDMO_CALCULATORS: Final[Mapping[str, BaseSDMOCalculator]] = MappingProxyType(
    {
        "sample_entropy": SampleEntropy(dim=2, r=0.15, acc_columns=["acc_is"]),
        "harmonic_ratio": HarmonicRatio(acc_columns=["acc_is", "acc_pa"]),
        "sd_range": SDRange(),
        "log_dimensionless_jerk": LogDimensionlessJerk(
            acc_columns=["acc_is", "acc_ml", "acc_pa"], gyr_columns=["gyr_is", "gyr_ml", "gyr_pa"]
        ),
        "rms_jerk_ratio": RMSJerkRatio(),
        "freq_amp_width": FrequencyAmplitudeWidth(acc_columns=["acc_is", "acc_ml", "acc_pa"]),
        "regularity_symmetry": RegularitySymmetry(),
        "rms": RMS(acc_columns=["acc_is", "acc_ml", "acc_pa"], gyr_columns=["gyr_is", "gyr_ml", "gyr_pa"]),
        "stride_level": StrideLevelSDMO(
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
