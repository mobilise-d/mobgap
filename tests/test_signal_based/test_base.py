from mobgap.signal_based import (
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
from mobgap.signal_based.base import BaseSDMOCalculator


def test_all_algorithms_inherit_base():
    algorithms = [
        SampleEntropy,
        HarmonicRatio,
        SDRange,
        Jerk,
        FrequencyAmplitudeWidthSlope,
        RegularitySymmetry,
        RMS,
        StrideLevelSDMO,
        TurnSDMO,
    ]
    for algo in algorithms:
        assert issubclass(algo, BaseSDMOCalculator), f"{algo.__name__} does not inherit from BaseSDMOCalculator"


def test_all_algorithms_have_calculate_method():
    algorithms = [
        SampleEntropy,
        HarmonicRatio,
        SDRange,
        Jerk,
        FrequencyAmplitudeWidthSlope,
        RegularitySymmetry,
        RMS,
        StrideLevelSDMO,
        TurnSDMO,
    ]
    for algo in algorithms:
        assert hasattr(algo, "calculate"), f"{algo.__name__} missing calculate method"
        assert callable(algo.calculate), f"{algo.__name__}.calculate is not callable"
