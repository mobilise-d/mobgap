import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.secondary_outcomes import (
    SpectralPeakFeatures,
)


class TestMetaSpectralPeakFeatures(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = SpectralPeakFeatures

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS(acc_columns=["acc_is", "acc_ml", "acc_pa"]).calculate(
            data, sampling_rate_hz=sampling_rate_hz
        )


class TestSpectralPeakFeatures:
    def test_hamming_tone_height_frequency_and_interpolated_half_height_width(self):
        t = np.arange(1024) / 64
        data = pd.DataFrame({"acc_is": 7 + 100 * np.sin(2 * np.pi * 2 * t)})
        row = SpectralPeakFeatures(acc_columns=["acc_is"]).calculate(data, sampling_rate_hz=64).signal_based_parameters_
        # Independent finite Hamming-window Fourier sums at 1.5/1.75/2/2.25/2.5 Hz.
        # Density height also follows 2*(.54**2/(.54**2+.46**2/2))*(1023/1024).
        assert row.freq_is == 2
        assert row.amplitude_is == pytest.approx(1.466105859964771)
        assert row.width_is == pytest.approx(0.6729226294134439)

    def test_configured_ml_axis_without_vertical_axis_and_out_of_band_tone(self):
        t = np.arange(1024) / 64
        data = pd.DataFrame({"acc_ml": np.sin(2 * np.pi * t) + 10 * np.sin(2 * np.pi * 4 * t)})
        row = SpectralPeakFeatures(acc_columns=["acc_ml"]).calculate(data, sampling_rate_hz=64).signal_based_parameters_
        assert row.freq_ml == 1
        assert "freq_is" not in row

    def test_local_peak_on_search_boundary_retains_frequency_with_undefined_width(self):
        t = np.arange(1024) / 64
        row = (
            SpectralPeakFeatures(acc_columns=["acc_is"], freq_band_is=(2.0, 3.0))
            .calculate(pd.DataFrame({"acc_is": np.sin(2 * np.pi * 2 * t)}), sampling_rate_hz=64)
            .signal_based_parameters_
        )
        assert row.freq_is == 2
        assert np.isnan(row.width_is)

    @pytest.mark.parametrize("values", [[], [1.0], [1.0] * 128, [0.1] * 1000, [np.nan] * 128])
    def test_undefined_spectrum_is_explicit(self, values):
        out = (
            SpectralPeakFeatures(acc_columns=["acc_is"])
            .calculate(pd.DataFrame({"acc_is": values}), sampling_rate_hz=64)
            .signal_based_parameters_
        )
        assert out.shape == (3,)
        assert out.isna().all().all()
