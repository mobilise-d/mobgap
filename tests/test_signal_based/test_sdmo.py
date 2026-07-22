import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS
from mobgap.data import LabExampleDataset
from mobgap.data_transform import Resample
from mobgap.signal_based import (
    RMS,
    AngularAcceleration,
    FrequencyAmplitudeWidth,
    HarmonicRatio,
    Jerk,
    RegularitySymmetry,
    SampleEntropy,
    SDRange,
    StrideLevelSDMO,
    TurnSDMO,
)
from mobgap.utils.conversions import to_body_frame


@pytest.fixture
def example_walking_bout():
    data = LabExampleDataset(reference_system="INDIP").get_subset(test="Test11")[0]
    wb_id = data.reference_parameters_relative_to_wb_.turn_parameters.index.get_level_values(0)[0]
    turn_list = data.reference_parameters_relative_to_wb_.turn_parameters.loc[wb_id]
    stride_list = data.reference_parameters_relative_to_wb_.stride_parameters.loc[wb_id].rename(
        columns={"duration_s": "stride_duration_s", "length_m": "stride_length_m"}
    )
    stride_list["cadence_spm"] = 60 * stride_list["speed_mps"] / stride_list["stride_length_m"]
    sampling_rate_hz = data.sampling_rate_hz
    data_start = data.reference_parameters_.stride_parameters.loc[wb_id][["start", "end"]].min().min()
    data_end = data.reference_parameters_.stride_parameters.loc[wb_id][["start", "end"]].max().max()
    data = to_body_frame(data.data_ss.iloc[data_start:data_end])
    return data, stride_list, turn_list, sampling_rate_hz


class TestMetaTurnSDMO(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = TurnSDMO

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS().calculate(data, sampling_rate_hz=sampling_rate_hz, turn_list=turn_list)


class TestMetaStrideLevelSDMO(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = StrideLevelSDMO

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS(
            stride_list_columns=["stride_length_m", "cadence_spm", "stride_duration_s"]
        ).calculate(data, stride_list=stride_list)


class TestMetaRMS(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = RMS

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS().calculate(data)


class TestMetaRegularitySymmetry(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = RegularitySymmetry

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS().calculate(data, sampling_rate_hz=sampling_rate_hz, replicate_matlab=True)


class TestMetaFrequencyAmplitudeWidthSlope(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = FrequencyAmplitudeWidth

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS(acc_columns=["acc_is", "acc_ml", "acc_pa"]).calculate(
            data, sampling_rate_hz=sampling_rate_hz
        )


class TestMetaSampleEntropy(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = SampleEntropy

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS(acc_columns=["acc_is"]).calculate(data, sampling_rate_hz=sampling_rate_hz)


class TestMetaHarmonicRatio(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = HarmonicRatio

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS(acc_columns=["acc_is"]).calculate(
            data, stride_list=stride_list, sampling_rate_hz=sampling_rate_hz
        )


class TestMetaSDRange(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = SDRange

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS().calculate(data)


class TestMetaJerk(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = Jerk

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS(acc_columns=["acc_is", "acc_ml", "acc_pa"]).calculate(
            data, sampling_rate_hz=sampling_rate_hz
        )


class TestMetaAngularAcceleration(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = AngularAcceleration

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS(gyr_columns=["gyr_is", "gyr_ml", "gyr_pa"]).calculate(
            data, sampling_rate_hz=sampling_rate_hz
        )


@pytest.mark.parametrize(
    "algorithm",
    [
        TurnSDMO(),
        StrideLevelSDMO(),
        RMS(),
        RegularitySymmetry(),
        FrequencyAmplitudeWidth(),
        SampleEntropy(),
        HarmonicRatio(),
        SDRange(),
        Jerk(),
        AngularAcceleration(),
    ],
)
def test_result_attributes_are_created_by_calculate(algorithm):
    assert not hasattr(algorithm, "signal_based_parameters_")


class TestTurnSDMO:
    def test_result_is_created_by_calculate(self):
        algo = TurnSDMO()

        assert not hasattr(algo, "signal_based_parameters_")

        result = algo.calculate(
            pd.DataFrame(np.random.randn(100, 6), columns=BF_SENSOR_COLS),
            sampling_rate_hz=100,
            turn_list=pd.DataFrame(),
        )
        assert result.signal_based_parameters_.empty

    def test_no_turns(self):
        algo = TurnSDMO()
        result = algo.calculate(
            pd.DataFrame(np.random.randn(1000, 6), columns=BF_SENSOR_COLS),
            sampling_rate_hz=100,
            turn_list=pd.DataFrame([]),
        )
        assert result is algo
        assert result.signal_based_parameters_.empty

    def test_pipe_data(self):
        data = [[0, 20, 72, 0.51, -52.0, "right"]]
        columns = ["turn_id", "start", "end", "duration_s", "angle_deg", "direction"]
        turn_list = pd.DataFrame(data, columns=columns).set_index("turn_id")
        algo = TurnSDMO()
        result = algo.calculate(
            pd.DataFrame(np.random.randn(100, 6), columns=BF_SENSOR_COLS), sampling_rate_hz=100, turn_list=turn_list
        )
        row = result.signal_based_parameters_.iloc[0]
        assert not np.isnan(row["turn_mean_ang_vel"])
        assert not np.isnan(row["turn_peak_ang_vel"])
        assert row["turn_smoothness"] >= 0
        assert 0 <= row["turn_dur_percentage_from_wb_dur"] <= 100


class TestStrideLevelSDMO:
    def test_calculates_coefficient_of_variation_as_percentage(self):
        stride_list = pd.DataFrame({"stride_duration_s": [1.0, 2.0, 3.0]})

        result = StrideLevelSDMO(stride_list_columns=["stride_duration_s"]).calculate(
            pd.DataFrame(), stride_list=stride_list
        )

        assert result.signal_based_parameters_.loc[0, "cv_stride_duration_s"] == pytest.approx(50.0)

    def test_no_columns(self):
        algo = StrideLevelSDMO(stride_list_columns=None)
        result = algo.calculate(
            pd.DataFrame(np.random.randn(1000, 6), columns=BF_SENSOR_COLS), stride_list=pd.DataFrame([0, 101, 1.0])
        )
        assert result is algo
        assert result.signal_based_parameters_.empty

    def test_missing_columns_warns(self):
        stride_list = pd.DataFrame([[0, 91, 0.9]], columns=["start", "end", "stride_duration_s"])
        algo = StrideLevelSDMO(stride_list_columns=["non_existent"])
        with pytest.warns(UserWarning, match="None of .* is available"):
            result = algo.calculate(
                pd.DataFrame(np.random.randn(1000, 6), columns=BF_SENSOR_COLS), stride_list=stride_list
            )
        assert result.signal_based_parameters_.empty

    def test_pipe_data(self):
        algo = StrideLevelSDMO(stride_list_columns=["stride_length_m", "cadence_spm", "stride_duration_s"])
        stride_list = pd.DataFrame(
            {
                "start": [0, 54],
                "end": [91, 185],
                "stride_duration_s": [0.9, 1.3],
                "stride_length_m": [1.2, 1.1],
                "speed_mps": [1.3, 0.8],
                "stance_time_s": [0.6, 0.9],
                "swing_time_s": [0.3, 0.4],
                "lr_label": ["left", "right"],
                "cadence_spm": [45, 35],
            }
        )
        result = algo.calculate(pd.DataFrame(np.random.randn(200, 6), columns=BF_SENSOR_COLS), stride_list=stride_list)
        for col in ["cv_stride_length_m", "cv_cadence_spm", "cv_stride_duration_s"]:
            assert result.signal_based_parameters_.iloc[0][col] > 0


class TestRMS:
    def test_gyroscope_rms_is_available_without_acceleration(self):
        data = pd.DataFrame({"gyr_is": [3.0, 3.0], "gyr_ml": [-4.0, -4.0]})

        result = RMS().calculate(data).signal_based_parameters_

        expected = pd.DataFrame({"rms_gyr_is": [3.0], "rms_gyr_ml": [4.0]})
        pd.testing.assert_frame_equal(result, expected)

    def test_constant_signal(self):
        algo = RMS()
        result = algo.calculate(pd.DataFrame(np.ones((100, 6)), columns=BF_SENSOR_COLS))
        df = result.signal_based_parameters_

        for col in ["rms_acc_is", "rms_acc_ml", "rms_acc_pa"]:
            assert df[col].iloc[0] == pytest.approx(0.0, abs=1e-12)
        for col in ["rms_gyr_is", "rms_gyr_ml", "rms_gyr_pa"]:
            assert df[col].iloc[0] == pytest.approx(1.0, abs=1e-12)

    def test_acceleration_total_and_ratios_exclude_gyroscope(self):
        acc_is = np.array([1.0, -1.0, -1.0, 1.0])
        data = pd.DataFrame(
            {
                "acc_is": acc_is,
                "acc_ml": 2 * acc_is,
                "acc_pa": np.zeros_like(acc_is),
                "gyr_is": np.full_like(acc_is, 10.0),
                "gyr_ml": np.full_like(acc_is, 20.0),
                "gyr_pa": np.full_like(acc_is, 30.0),
            }
        )

        result = RMS().calculate(data).signal_based_parameters_.iloc[0]

        expected_total = np.sqrt(5)
        assert result["rms_total_acc"] == pytest.approx(expected_total)
        assert result["rms_ratio_acc_is"] == pytest.approx(1 / expected_total)
        assert result["rms_ratio_acc_ml"] == pytest.approx(2 / expected_total)
        assert result["rms_ratio_acc_pa"] == pytest.approx(0)

    def test_acceleration_rms_removes_only_dc_offset(self):
        data = pd.DataFrame({"acc_is": [10.0, 11.0, 12.0]})

        result = RMS().calculate(data).signal_based_parameters_.iloc[0]

        expected_rms = np.sqrt(2 / 3)
        assert result["rms_acc_is"] == pytest.approx(expected_rms)
        assert result["rms_total_acc"] == pytest.approx(expected_rms)
        assert result["rms_ratio_acc_is"] == pytest.approx(1)

    def test_unrelated_columns_are_ignored(self):
        data = pd.DataFrame({"acc_is": [1.0, -1.0], "temperature": [20.0, 22.0]})

        result = RMS().calculate(data).signal_based_parameters_

        assert list(result.columns) == ["rms_acc_is", "rms_total_acc", "rms_ratio_acc_is"]

    def test_signal_with_dc_offset(self):
        t = np.linspace(0, 1, 200)
        A = 2.0
        dc = 5.0
        signal = dc + A * np.sin(2 * np.pi * 5 * t)
        data = pd.DataFrame({"acc_is": signal})
        algo = RMS()
        result = algo.calculate(data)
        df = result.signal_based_parameters_

        expected = A / np.sqrt(2)
        assert df["rms_acc_is"].iloc[0] == pytest.approx(expected, rel=1e-1)
        assert df["rms_total_acc"].iloc[0] == pytest.approx(expected, rel=1e-1)
        assert df["rms_ratio_acc_is"].iloc[0] == pytest.approx(1.0, rel=1e-2)


class TestRegularitySymmetry:
    def test_missing_columns_returns_self(self):
        data_missing = pd.DataFrame({"acc_is": np.random.randn(100)})
        algo = RegularitySymmetry()
        result = algo.calculate(data_missing, sampling_rate_hz=100, replicate_matlab=False)
        assert result is algo
        assert result.signal_based_parameters_.empty

    def test_pipe_data(self):
        algo = RegularitySymmetry()
        result = algo.calculate(
            pd.DataFrame(np.random.randn(200, 6), columns=BF_SENSOR_COLS), sampling_rate_hz=100, replicate_matlab=False
        )
        df = result.signal_based_parameters_

        expected = [
            "step_regularity_is",
            "stride_regularity_is",
            "asymmetry_mn_is",
            "symmetry_k_is",
            "asymmetry_g_is",
            "step_regularity_ml",
            "stride_regularity_ml",
            "step_regularity_pa",
            "stride_regularity_pa",
        ]
        assert set(df.columns) == set(expected)

        row = df.iloc[0]

        for col in [
            "step_regularity_is",
            "stride_regularity_is",
            "step_regularity_ml",
            "stride_regularity_ml",
            "step_regularity_pa",
            "stride_regularity_pa",
        ]:
            val = row[col]
            assert not np.isnan(val), f"{col} is NaN"
            assert 0 <= val <= 1.0, f"{col} = {val} outside [0, 1.0]"

        assert not np.isnan(row["asymmetry_mn_is"])
        assert row["asymmetry_mn_is"] >= 0

        assert not np.isnan(row["symmetry_k_is"])
        assert row["symmetry_k_is"] >= 0

        assert not np.isnan(row["asymmetry_g_is"])


class TestFrequencyAmplitudeWidthSlope:
    def test_pipe_data(self):
        algo = FrequencyAmplitudeWidth(acc_columns=["acc_is", "acc_ml", "acc_pa"])
        result = algo.calculate(pd.DataFrame(np.random.randn(200, 6), columns=BF_SENSOR_COLS), sampling_rate_hz=100)
        df = result.signal_based_parameters_
        expected_cols = [
            "amplitude_is",
            "amplitude_ml",
            "amplitude_pa",
            "freq_is",
            "freq_ml",
            "freq_pa",
            "width_is",
            "width_ml",
            "width_pa",
        ]
        assert set(df.columns) == set(expected_cols)
        assert not df.isnull().all().all()


class TestSampleEntropy:
    def test_missing_columns_returns_self(self):
        algo = SampleEntropy(acc_columns=["acc_x"])
        result = algo.calculate(
            pd.DataFrame(np.random.randn(500, 6), columns=BF_SENSOR_COLS),
            sampling_rate_hz=100,
        )
        assert result is algo
        assert result.signal_based_parameters_.empty

    def test_insufficient_samples_returns_nan(self):
        algo = SampleEntropy(acc_columns=["acc_is"], num_samples_threshold=200)
        result = algo.calculate(
            pd.DataFrame(np.random.randn(100, 6), columns=BF_SENSOR_COLS),
            sampling_rate_hz=100,
        )
        df = result.signal_based_parameters_
        assert df.shape == (1, 1)
        assert np.isnan(df.iloc[0, 0])

    def test_empty_input_returns_nan(self):
        result = SampleEntropy(acc_columns=["acc_is"]).calculate(pd.DataFrame({"acc_is": []}), sampling_rate_hz=100)

        assert np.isnan(result.signal_based_parameters_.loc[0, "sample_entropy_acc_is"])

    def test_single_sample_input_returns_nan(self):
        result = SampleEntropy(acc_columns=["acc_is"]).calculate(pd.DataFrame({"acc_is": [1.0]}), sampling_rate_hz=100)

        assert np.isnan(result.signal_based_parameters_.loc[0, "sample_entropy_acc_is"])

    def test_sample_threshold_is_applied_per_axis(self):
        data = pd.DataFrame(
            np.random.default_rng(0).normal(size=(300, 2)),
            columns=["acc_is", "acc_ml"],
        )

        result = SampleEntropy(acc_columns=["acc_is", "acc_ml"], num_samples_threshold=200).calculate(
            data, sampling_rate_hz=100
        )

        assert result.signal_based_parameters_.isna().all().all()

    def test_entropy_is_independent_of_other_requested_axes(self):
        data = pd.DataFrame(
            np.random.default_rng(0).normal(size=(800, 2)),
            columns=["acc_is", "acc_ml"],
        )
        kwargs = {"sampling_rate_hz": 100}

        entropy_single_axis = (
            SampleEntropy(acc_columns=["acc_is"])
            .calculate(data, **kwargs)
            .signal_based_parameters_["sample_entropy_acc_is"]
        )
        entropy_multiple_axes = (
            SampleEntropy(acc_columns=["acc_is", "acc_ml"])
            .calculate(data, **kwargs)
            .signal_based_parameters_["sample_entropy_acc_is"]
        )

        pd.testing.assert_series_equal(entropy_single_axis, entropy_multiple_axes)

    def test_resamples_to_internal_sampling_rate(self):
        data_100_hz = pd.DataFrame({"acc_is": np.random.default_rng(0).normal(size=800)})
        data_50_hz = (
            Resample(target_sampling_rate_hz=50, attempt_index_resample=False)
            .transform(data_100_hz, sampling_rate_hz=100)
            .transformed_data_
        )
        algorithm = SampleEntropy(
            acc_columns=["acc_is"],
            internal_sampling_rate_hz=50,
            num_samples_threshold=100,
        )

        entropy_from_100_hz = algorithm.clone().calculate(data_100_hz, sampling_rate_hz=100).signal_based_parameters_
        entropy_from_50_hz = algorithm.clone().calculate(data_50_hz, sampling_rate_hz=50).signal_based_parameters_

        pd.testing.assert_frame_equal(entropy_from_100_hz, entropy_from_50_hz)

    def test_sufficient_samples_returns_float(self):
        algo = SampleEntropy(acc_columns=["acc_is"], num_samples_threshold=200)
        result = algo.calculate(
            pd.DataFrame(np.random.randn(500, 6), columns=BF_SENSOR_COLS),
            sampling_rate_hz=100,
        )
        df = result.signal_based_parameters_
        assert df.shape == (1, 1)
        val = df.iloc[0, 0]
        assert isinstance(val, float) and not np.isnan(val) and val > 0


class TestHarmonicRatio:
    def test_insufficient_strides_returns_self(self):
        stride_list = pd.DataFrame(
            {
                "start": [0, 60],
                "end": [101, 211],
                "stride_duration_s": [1.0, 1.5],
                "stride_length_m": [0.9, 1.1],
                "speed_mps": [0.8, 0.6],
                "stance_time_s": [0.8, 0.9],
                "swing_time_s": [0.2, 0.6],
                "lr_label": ["left", "right"],
                "cadence_spm": [45, 35],
            }
        )
        algo = HarmonicRatio(acc_columns=["acc_is"])
        result = algo.calculate(
            pd.DataFrame(np.random.randn(250, 6), columns=BF_SENSOR_COLS), stride_list=stride_list, sampling_rate_hz=100
        )
        assert result.signal_based_parameters_.empty

    def test_sufficient_strides_returns_float(self):
        stride_list = pd.DataFrame(
            {
                "start": [0, 60, 120],
                "end": [101, 211, 281],
                "stride_duration_s": [1.0, 1.5, 1.6],
                "stride_length_m": [0.9, 1.1, 1.0],
                "speed_mps": [0.8, 0.6, 0.7],
                "stance_time_s": [0.8, 0.9, 0.9],
                "swing_time_s": [0.2, 0.6, 0.7],
                "lr_label": ["left", "right", "left"],
                "cadence_spm": [45, 35, 50],
            }
        )
        algo = HarmonicRatio(acc_columns=["acc_is"])
        result = algo.calculate(
            pd.DataFrame(np.random.randn(300, 6), columns=BF_SENSOR_COLS), stride_list=stride_list, sampling_rate_hz=100
        )
        df = result.signal_based_parameters_
        assert df.shape == (1, 1)
        val = df.iloc[0, 0]
        assert isinstance(val, float) and not np.isnan(val)


class TestSDRange:
    def test_output_has_correct_columns(self):
        algo = SDRange()
        data = pd.DataFrame(np.random.randn(300, 6), columns=BF_SENSOR_COLS)
        result = algo.calculate(data)
        df = result.signal_based_parameters_
        expected_cols = [f"sd_{c}" for c in data.columns] + [f"range_{c}" for c in data.columns if "acc" in c]
        assert set(df.columns) == set(expected_cols)
        assert df.shape == (1, len(expected_cols))


class TestJerk:
    def test_linear_acceleration_has_constant_jerk(self):
        sampling_rate_hz = 10.0
        time = np.arange(11) / sampling_rate_hz
        data = pd.DataFrame({"acc_is": 2.0 * time})

        result = Jerk(acc_columns=["acc_is"]).calculate(data, sampling_rate_hz=sampling_rate_hz)

        assert result.signal_based_parameters_.loc[0, "jerk_acc_is"] == pytest.approx(2.0)

    def test_requires_acc_columns(self):
        algo = Jerk(acc_columns=["acc_is", "acc_ml", "acc_pa"])
        result = algo.calculate(pd.DataFrame(np.random.randn(300, 6), columns=BF_SENSOR_COLS), sampling_rate_hz=100)
        df = result.signal_based_parameters_
        expected = [f"jerk_{col}" for col in algo.acc_columns]
        assert set(df.columns) == set(expected)
        assert all(not np.isnan(value) and value > 0 for value in df.iloc[0])


class TestAngularAcceleration:
    def test_linear_angular_velocity_has_constant_angular_acceleration(self):
        sampling_rate_hz = 10.0
        time = np.arange(11) / sampling_rate_hz
        data = pd.DataFrame({"gyr_is": 3.0 * time})

        result = AngularAcceleration(gyr_columns=["gyr_is"]).calculate(data, sampling_rate_hz=sampling_rate_hz)

        assert result.signal_based_parameters_.loc[0, "angular_acceleration_gyr_is"] == pytest.approx(3.0)

    def test_requires_gyr_columns(self):
        algo = AngularAcceleration(gyr_columns=["gyr_is", "gyr_ml", "gyr_pa"])
        result = algo.calculate(pd.DataFrame(np.random.randn(300, 6), columns=BF_SENSOR_COLS), sampling_rate_hz=100)
        df = result.signal_based_parameters_
        expected = [f"angular_acceleration_{column}" for column in algo.gyr_columns]
        assert set(df.columns) == set(expected)
        assert all(not np.isnan(value) and value > 0 for value in df.iloc[0])
