"""Compatibility imports for secondary mobility outcome calculators."""

from mobgap.secondary_outcomes import (
    DEFAULT_WB_SDMO_CALCULATORS,
    HarmonicRatioFourier,
    LogDimensionlessJerk,
    SampleEntropy,
)
from mobgap.secondary_outcomes import GaitRegularityAndSymmetry as RegularitySymmetry
from mobgap.secondary_outcomes import HarmonicRatioPeakTrough as HarmonicRatio
from mobgap.secondary_outcomes import JerkRMSRatio as RMSJerkRatio
from mobgap.secondary_outcomes import RootMeanSquare as RMS  # noqa: N817 - compatibility alias
from mobgap.secondary_outcomes import SpectralPeakFeatures as FrequencyAmplitudeWidth
from mobgap.secondary_outcomes import StandardDeviationAndRange as SDRange
from mobgap.secondary_outcomes import StrideParameterCV as StrideLevelSDMO
from mobgap.secondary_outcomes import TurnKinematics as TurnSDMO

__all__ = [
    "DEFAULT_WB_SDMO_CALCULATORS",
    "RMS",
    "FrequencyAmplitudeWidth",
    "HarmonicRatio",
    "HarmonicRatioFourier",
    "LogDimensionlessJerk",
    "RMSJerkRatio",
    "RegularitySymmetry",
    "SDRange",
    "SampleEntropy",
    "StrideLevelSDMO",
    "TurnSDMO",
]
