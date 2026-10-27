import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.signal_based import HarmonicRatioFourier


def test_known_harmonics_use_supplied_boundaries_with_nonzero_offset():
    phase = np.arange(100) / 100
    stride = np.sin(4 * np.pi * phase) + 0.4 * np.sin(6 * np.pi * phase)
    # Nonfinite surrounding samples must not enter the stride's spectrum.
    data = pd.DataFrame({"acc_is": np.r_[np.full(23, np.nan), stride, np.full(17, np.nan)]})
    strides = pd.DataFrame({"start": [23], "end": [123]})
    result = HarmonicRatioFourier(acc_columns=["acc_is"]).calculate(data, stride_list=strides, sampling_rate_hz=100)
    assert result.signal_based_parameters_.harmonic_ratio_fourier_acc_is == pytest.approx(2.5)


@pytest.mark.parametrize("axis,expected", [("acc_is", 3.75), ("acc_pa", 3.75), ("acc_ml", 0.3)])
def test_mean_of_stride_ratios_with_different_lengths_gaps_scales_and_row_order(axis, expected):
    first = np.arange(100) / 100
    second = np.arange(125) / 125
    a = np.sin(4 * np.pi * first) + 0.4 * np.sin(6 * np.pi * first)
    b = 10 * (np.sin(4 * np.pi * second + 0.7) + 0.2 * np.sin(6 * np.pi * second - 0.3))
    signal = np.r_[np.full(23, np.nan), a, np.full(17, np.nan), b, np.full(9, np.nan)]
    strides = pd.DataFrame(
        {"start": [140, 23], "end": [265, 123], "lr_label": ["right", "left"], "stride_duration_s": [999.0, 999.0]},
        index=["second", "first"],
    )
    row = (
        HarmonicRatioFourier(acc_columns=[axis])
        .calculate(pd.DataFrame({axis: signal}), stride_list=strides, sampling_rate_hz=100)
        .signal_based_parameters_
    )
    # IS/PA ratios are 5 and 2.5, ML ratios are .2 and .4. Average ratios, not coefficients.
    assert row[f"harmonic_ratio_fourier_{axis}"] == pytest.approx(expected)


@pytest.mark.parametrize("length", [41, 64, 100, 151])
@pytest.mark.parametrize("sampling_rate_hz", [50, 100, 128])
def test_fourier_bins_follow_each_stride_period(length, sampling_rate_hz):
    phase = np.arange(length) / length
    signal = 9.81 + np.sin(4 * np.pi * phase + 0.3) + 0.4 * np.sin(6 * np.pi * phase - 1.2)
    row = (
        HarmonicRatioFourier(acc_columns=["acc_is"])
        .calculate(
            pd.DataFrame({"acc_is": signal}),
            stride_list=pd.DataFrame({"start": [0], "end": [length]}),
            sampling_rate_hz=sampling_rate_hz,
        )
        .signal_based_parameters_
    )
    assert row.harmonic_ratio_fourier_acc_is == pytest.approx(2.5)


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros(100),
        np.full(100, 0.1),
        np.full(100, np.nan),
        np.full(100, np.inf),
        np.sin(4 * np.pi * np.arange(40) / 40) + 0.2 * np.sin(6 * np.pi * np.arange(40) / 40),
    ],
)
def test_undefined_stride_axes_are_omitted_without_changing_valid_stride_ratio(invalid):
    phase = np.arange(100) / 100
    valid = np.sin(4 * np.pi * phase) + 0.4 * np.sin(6 * np.pi * phase)
    signal = np.r_[invalid, valid]
    strides = pd.DataFrame({"start": [0, len(invalid)], "end": [len(invalid), len(signal)]})
    row = (
        HarmonicRatioFourier(acc_columns=["acc_is", "acc_pa", "acc_ml"])
        .calculate(
            pd.DataFrame({"acc_is": signal, "acc_ml": np.zeros(len(signal))}), stride_list=strides, sampling_rate_hz=100
        )
        .signal_based_parameters_
    )
    assert row.harmonic_ratio_fourier_acc_is == pytest.approx(2.5)
    assert np.isnan(row.harmonic_ratio_fourier_acc_ml)
    assert "harmonic_ratio_fourier_acc_pa" not in row


def test_empty_strides_or_no_configured_available_axes_produce_no_outputs():
    data = pd.DataFrame({"acc_is": np.ones(100)})
    strides = pd.DataFrame({"start": [0], "end": [100]})
    for calculator, supplied in [
        (HarmonicRatioFourier(), strides),
        (HarmonicRatioFourier(acc_columns=["acc_pa"]), strides),
        (HarmonicRatioFourier(acc_columns=["acc_is"]), strides.iloc[:0]),
    ]:
        result = calculator.calculate(data, stride_list=supplied, sampling_rate_hz=100)
        assert result.signal_based_parameters_.empty


class TestMetaHarmonicRatioFourier(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = HarmonicRatioFourier

    @pytest.fixture
    def after_action_instance(self):
        phase = np.arange(100) / 100
        return self.ALGORITHM_CLASS(acc_columns=["acc_is"]).calculate(
            pd.DataFrame({"acc_is": np.sin(4 * np.pi * phase) + 0.4 * np.sin(6 * np.pi * phase)}),
            stride_list=pd.DataFrame({"start": [0], "end": [100]}),
            sampling_rate_hz=100,
        )


@pytest.mark.parametrize("axis,expected", [("acc_is", np.inf), ("acc_ml", 0.0)])
def test_exact_missing_harmonic_group_has_unclipped_ratio(axis, expected):
    # Two identical half-strides have exactly zero odd DFT coefficients.
    signal = np.tile(np.r_[1.0, np.zeros(31)], 2)
    row = (
        HarmonicRatioFourier(acc_columns=[axis])
        .calculate(
            pd.DataFrame({axis: signal}), stride_list=pd.DataFrame({"start": [0], "end": [64]}), sampling_rate_hz=100
        )
        .signal_based_parameters_
    )
    assert row[f"harmonic_ratio_fourier_{axis}"] == expected


def test_named_calculator_loop_calculates_both_estimators_without_overwriting_either():
    from mobgap.signal_based import HarmonicRatio

    phase = np.arange(100) / 100
    data = pd.DataFrame({"acc_is": np.sin(4 * np.pi * phase) + 0.4 * np.sin(6 * np.pi * phase)})
    strides = pd.DataFrame({"start": [0], "end": [100]})
    calculators = {
        "harmonic_ratio": HarmonicRatio(acc_columns=["acc_is"]),
        "harmonic_ratio_fourier": HarmonicRatioFourier(acc_columns=["acc_is"]),
    }
    row = pd.concat(
        {
            name: calculator.clone().calculate(data, stride_list=strides, sampling_rate_hz=100).signal_based_parameters_
            for name, calculator in calculators.items()
        },
        names=["algorithm", "parameter"],
    )
    assert row["harmonic_ratio_fourier", "harmonic_ratio_fourier_acc_is"] == pytest.approx(2.5)
    assert row["harmonic_ratio", "harmonic_ratio_acc_is"] == pytest.approx(25.195295004095286)
