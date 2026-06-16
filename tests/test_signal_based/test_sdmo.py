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
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data() -> pd.DataFrame:
    t = np.linspace(0, 10, 1000)
    acc_is = 1.0 + 0.5 * np.sin(2 * np.pi * 1.0 * t)
    acc_ml = 0.5 + 0.3 * np.sin(2 * np.pi * 2.0 * t)
    acc_pa = -0.5 + 0.4 * np.cos(2 * np.pi * 1.5 * t)
    gyr_is = 0.0 + 0.1 * np.sin(2 * np.pi * 0.8 * t)
    gyr_ml = 0.0 + 0.2 * np.cos(2 * np.pi * 1.2 * t)
    gyr_pa = 0.0 + 0.15 * np.sin(2 * np.pi * 0.9 * t)
    return pd.DataFrame({
        "acc_is": acc_is,
        "acc_ml": acc_ml,
        "acc_pa": acc_pa,
        "gyr_is": gyr_is,
        "gyr_ml": gyr_ml,
        "gyr_pa": gyr_pa,
    })

@pytest.fixture
def stride_list() -> pd.DataFrame:
    return pd.DataFrame({
        "start": [0, 200, 400, 600, 800],
        "end": [199, 399, 599, 799, 999],
        "stride_length_m": [1.2, 1.3, 1.1, 1.4, 1.2],
        "cadence_spm": [110, 108, 112, 109, 111],
        "stride_duration_s": [1.0, 0.98, 1.02, 0.99, 1.01],
    })

@pytest.fixture
def turn_list() -> pd.DataFrame:
    return pd.DataFrame({
        "start": [50, 700],
        "end": [150, 850],
        "duration_s": [1.0, 1.5],
    })

@pytest.fixture
def example_walking_bout(sample_data, stride_list, turn_list):
    return {
        "data": sample_data,
        "stride_list": stride_list,
        "turn_list": turn_list,
        "sampling_rate_hz": 100.0,
    }

class TestSampleEntropy:
    def test_missing_columns_returns_self(self, sample_data):
        algo = SampleEntropy(acc_columns=["acc_x"])
        result = algo.calculate(sample_data)
        assert result is algo
        assert result.signal_based_parameters.empty

    def test_insufficient_samples_returns_nan(self, sample_data):
        # take only first 100 samples
        data_small = sample_data.iloc[:100]
        algo = SampleEntropy(acc_columns=["acc_is"], num_samples_threshold=200)
        result = algo.calculate(data_small)
        df = result.signal_based_parameters
        assert df.shape == (1, 1)
        assert np.isnan(df.iloc[0, 0])

    def test_sufficient_samples_returns_float(self, sample_data):
        algo = SampleEntropy(acc_columns=["acc_is"], num_samples_threshold=200)
        result = algo.calculate(sample_data)
        df = result.signal_based_parameters
        assert df.shape == (1, 1)
        val = df.iloc[0, 0]
        assert isinstance(val, float) and not np.isnan(val) and val > 0

class TestHarmonicRatio:
    def test_insufficient_strides_returns_self(self, sample_data, stride_list):
        sl = stride_list.iloc[:2]
        algo = HarmonicRatio(acc_columns=["acc_is"])
        result = algo.calculate(sample_data, stride_list=sl, sampling_rate_hz=100)
        assert result.signal_based_parameters.empty

    def test_sufficient_strides_returns_float(self, sample_data, stride_list):
        algo = HarmonicRatio(acc_columns=["acc_is"])
        result = algo.calculate(sample_data, stride_list=stride_list, sampling_rate_hz=100)
        df = result.signal_based_parameters
        assert df.shape == (1, 1)
        val = df.iloc[0, 0]
        assert isinstance(val, float) and not np.isnan(val)

class TestSDRange:
    def test_output_has_correct_columns(self, sample_data):
        algo = SDRange()
        result = algo.calculate(sample_data)
        df = result.signal_based_parameters
        expected_cols = [f"sd_{c}" for c in sample_data.columns] + \
                        [f"range_{c}" for c in sample_data.columns if "acc" in c]
        assert set(df.columns) == set(expected_cols)
        assert df.shape == (1, len(expected_cols))

class TestJerk:
    def test_requires_acc_columns(self, sample_data):
        algo = Jerk(acc_columns=["acc_is", "acc_ml", "acc_pa"], gyr_columns=["gyr_is", "gyr_ml", "gyr_pa"])
        result = algo.calculate(sample_data, sampling_rate_hz=100)
        df = result.signal_based_parameters
        expected = [f"jerk_{col}" for col in algo.acc_columns] + [f"jerk_{col}" for col in algo.gyr_columns]
        assert set(df.columns) == set(expected)
        assert all(not np.isnan(v) and v > 0 for v in df.iloc[0])

    def test_missing_gyr_columns_ignores(self, sample_data):
        # Remove gyr columns
        data_no_gyr = sample_data.drop(columns=["gyr_is", "gyr_ml", "gyr_pa"])
        algo = Jerk(acc_columns=["acc_is", "acc_ml", "acc_pa"], gyr_columns=["gyr_is", "gyr_ml", "gyr_pa"])
        result = algo.calculate(data_no_gyr, sampling_rate_hz=100)
        df = result.signal_based_parameters
        expected = [f"jerk_{col}" for col in algo.acc_columns]
        assert set(df.columns) == set(expected)

class TestRMS:
    def test_no_acc_columns_returns_self(self):
        data = pd.DataFrame({
            "gyr_is": np.random.randn(100),
            "gyr_ml": np.random.randn(100),
        })
        algo = RMS()
        result = algo.calculate(data)
        assert result is algo
        assert result.signal_based_parameters.empty

    def test_constant_signal(self):
        data = pd.DataFrame({
            "acc_is": 2.0 * np.ones(100),
            "acc_ml": 3.0 * np.ones(100),
            "acc_pa": 4.0 * np.ones(100),
        })
        algo = RMS()
        result = algo.calculate(data)
        df = result.signal_based_parameters

        for col in ["rms_acc_is", "rms_acc_ml", "rms_acc_pa"]:
            assert df[col].iloc[0] == pytest.approx(0.0, abs=1e-12)

        assert df["rms_total_acc"].iloc[0] == pytest.approx(0.0, abs=1e-12)

    def test_signal_with_dc_offset(self):
        t = np.linspace(0, 1, 1000)
        A = 2.0
        dc = 5.0
        signal = dc + A * np.sin(2 * np.pi * 5 * t)
        data = pd.DataFrame({"acc_is": signal})
        algo = RMS()
        result = algo.calculate(data)
        df = result.signal_based_parameters

        expected = A / np.sqrt(2)
        assert df["rms_acc_is"].iloc[0] == pytest.approx(expected, rel=1e-1)
        assert df["rms_total_acc"].iloc[0] == pytest.approx(expected, rel=1e-1)
        assert df["rms_ratio_acc_is"].iloc[0] == pytest.approx(1.0, rel=1e-2)


class TestFrequencyAmplitudeWidthSlope:
    def test_pipe_data(self, sample_data):
        algo = FrequencyAmplitudeWidthSlope(acc_columns=["acc_is", "acc_ml", "acc_pa"])
        result = algo.calculate(sample_data, sampling_rate_hz=100)
        df = result.signal_based_parameters
        expected_cols = ["amplitude_is", "amplitude_ml", "amplitude_pa",
                         "freq_is", "freq_ml", "freq_pa",
                         "width_is", "width_ml", "width_pa"]
        assert set(df.columns) == set(expected_cols)
        assert not df.isnull().all().all()

class TestRegularitySymmetry:
    def test_missing_columns_returns_self(self, sample_data):
        data_missing = sample_data[["acc_is"]]
        algo = RegularitySymmetry()
        result = algo.calculate(data_missing, sampling_rate_hz=100, replicate_matlab=False)
        assert result is algo
        assert result.signal_based_parameters.empty

    def test_pipe_data(self, sample_data):
        algo = RegularitySymmetry()
        result = algo.calculate(sample_data, sampling_rate_hz=100, replicate_matlab=False)
        df = result.signal_based_parameters

        expected = [
            "step_regularity_is", "stride_regularity_is",
            "asymmetry_mn_is", "symmetry_k_is", "asymmetry_g_is",
            "step_regularity_ml", "stride_regularity_ml",
            "step_regularity_pa", "stride_regularity_pa",
        ]
        assert set(df.columns) == set(expected)

        row = df.iloc[0]

        for col in ["step_regularity_is", "stride_regularity_is",
                    "step_regularity_ml", "stride_regularity_ml",
                    "step_regularity_pa", "stride_regularity_pa"]:
            val = row[col]
            assert not np.isnan(val), f"{col} is NaN"
            assert 0 <= val <= 1.03, f"{col} = {val} outside [0, 1.01]"

        assert not np.isnan(row["asymmetry_mn_is"])
        assert row["asymmetry_mn_is"] >= 0

        assert not np.isnan(row["symmetry_k_is"])
        assert row["symmetry_k_is"] >= 0

        assert not np.isnan(row["asymmetry_g_is"])

class TestStrideLevelSDMO:
    def test_no_columns_returns_self(self, sample_data, stride_list):
        algo = StrideLevelSDMO(stride_list_columns=None)
        result = algo.calculate(sample_data, stride_list=stride_list)
        assert result is algo
        assert result.signal_based_parameters.empty

    def test_missing_columns_warns(self, sample_data, stride_list):
        algo = StrideLevelSDMO(stride_list_columns=["non_existent"])
        with pytest.warns(UserWarning, match="None of .* is available"):
            result = algo.calculate(sample_data, stride_list=stride_list)
        assert result.signal_based_parameters.empty

    def test_pipe_data(self, sample_data, stride_list):
        algo = StrideLevelSDMO(stride_list_columns=["stride_length_m", "cadence_spm", "stride_duration_s"])
        result = algo.calculate(sample_data, stride_list=stride_list)
        df = result.signal_based_parameters
        expected = ["cv_stride_length_m", "cv_cadence_spm", "cv_stride_duration_s"]
        assert set(df.columns) == set(expected)
        for col in expected:
            assert df.iloc[0][col] > 0

class TestTurnSDMO:
    def test_no_turns_returns_self(self, sample_data, turn_list):
        turn_list_empty = pd.DataFrame(columns=["start", "end", "duration_s"])
        algo = TurnSDMO()
        result = algo.calculate(sample_data, sampling_rate_hz=100, turn_list=turn_list_empty)
        assert result is algo
        assert result.signal_based_parameters.empty

    def test_pipe_data(self, sample_data, turn_list):
        algo = TurnSDMO()
        result = algo.calculate(sample_data, sampling_rate_hz=100, turn_list=turn_list)
        df = result.signal_based_parameters
        expected = ["turn_mean_ang_vel", "turn_peak_ang_vel", "turn_smoothness", "turn_dur_percentage_from_wb_dur"]
        assert set(df.columns) == set(expected)

        row = df.iloc[0]
        assert not np.isnan(row["turn_mean_ang_vel"])
        assert not np.isnan(row["turn_peak_ang_vel"])
        assert row["turn_smoothness"] >= 0
        assert 0 <= row["turn_dur_percentage_from_wb_dur"] <= 100
