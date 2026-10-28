import pytest

from mobgap.secondary_outcomes import (
    GaitRegularityAndSymmetry,
    HarmonicRatioPeakTrough,
    JerkRMSRatio,
    LogDimensionlessJerk,
    RootMeanSquare,
    SampleEntropy,
    SpectralPeakFeatures,
    StandardDeviationAndRange,
    StrideParameterCV,
    TurnKinematics,
)


@pytest.mark.parametrize(
    "algorithm",
    [
        TurnKinematics(),
        StrideParameterCV(),
        RootMeanSquare(),
        GaitRegularityAndSymmetry(),
        SpectralPeakFeatures(),
        SampleEntropy(),
        HarmonicRatioPeakTrough(),
        StandardDeviationAndRange(),
        LogDimensionlessJerk(),
        JerkRMSRatio(),
    ],
)
def test_result_attributes_are_created_by_calculate(algorithm):
    assert not hasattr(algorithm, "signal_based_parameters_")
