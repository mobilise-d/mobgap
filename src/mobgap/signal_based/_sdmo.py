"""Compatibility imports for secondary outcome calculators and numerical helpers."""

from mobgap.secondary_outcomes import GaitRegularityAndSymmetry as RegularitySymmetry
from mobgap.secondary_outcomes import HarmonicRatioFourier, LogDimensionlessJerk, SampleEntropy
from mobgap.secondary_outcomes import HarmonicRatioPeakTrough as HarmonicRatio
from mobgap.secondary_outcomes import JerkRMSRatio as RMSJerkRatio
from mobgap.secondary_outcomes import RootMeanSquare as RMS  # noqa: N817 - compatibility alias
from mobgap.secondary_outcomes import SpectralPeakFeatures as FrequencyAmplitudeWidth
from mobgap.secondary_outcomes import StandardDeviationAndRange as SDRange
from mobgap.secondary_outcomes import StrideParameterCV as StrideLevelSDMO
from mobgap.secondary_outcomes import TurnKinematics as TurnSDMO
from mobgap.secondary_outcomes.complexity._sample_entropy import _count_template_matches
from mobgap.secondary_outcomes.periodicity._gait_regularity_and_symmetry import (
    _matlab_smooth_moving_ave,
    _pd_smooth_moving_ave,
)
from mobgap.secondary_outcomes.smoothness._utils import _integrated_squared_derivative, _log_dimensionless_jerk

__all__ = [
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
    "_count_template_matches",
    "_integrated_squared_derivative",
    "_log_dimensionless_jerk",
    "_matlab_smooth_moving_ave",
    "_pd_smooth_moving_ave",
]
