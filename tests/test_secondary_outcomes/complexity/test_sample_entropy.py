import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS
from mobgap.data_transform import Resample
from mobgap.secondary_outcomes import (
    SampleEntropy,
)


class TestMetaSampleEntropy(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = SampleEntropy

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS(acc_columns=["acc_is"]).calculate(data, sampling_rate_hz=sampling_rate_hz)


class TestSampleEntropy:
    @pytest.mark.parametrize(
        "values,expected",
        [
            ([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.log(3)),
            ([0.0, 0.0, 0.0, 1.0, 0.0], np.inf),
            ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], np.nan),
            ([2.0] * 6, 0.0),
        ],
    )
    @pytest.mark.parametrize("acc_columns", [["acc_is"], ["acc_is", "acc_ml"]])
    @pytest.mark.parametrize("dtype", [np.float64, np.float16])
    def test_exact_common_extendible_template_counts(self, values, expected, acc_columns, dtype):
        # For 000010: m=2 has 3 unordered matching pairs, m=3 has 1.
        # 00010 has one matching pair at m=2 and none at m=3.
        row = (
            SampleEntropy(acc_columns=acc_columns, r=0.15, num_samples_threshold=0)
            .calculate(pd.DataFrame({"acc_is": np.asarray(values, dtype=dtype)}), sampling_rate_hz=50)
            .signal_based_parameters_
        )
        if np.isnan(expected):
            assert np.isnan(row["sample_entropy_acc_is"])
        else:
            assert row["sample_entropy_acc_is"] == pytest.approx(expected)

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
        assert df.shape == (1,)
        assert np.isnan(df.iloc[0])

    def test_empty_input_returns_nan(self):
        result = SampleEntropy(acc_columns=["acc_is"]).calculate(pd.DataFrame({"acc_is": []}), sampling_rate_hz=100)

        assert np.isnan(result.signal_based_parameters_.loc["sample_entropy_acc_is"])

    def test_single_sample_input_returns_nan(self):
        result = SampleEntropy(acc_columns=["acc_is"]).calculate(pd.DataFrame({"acc_is": [1.0]}), sampling_rate_hz=100)

        assert np.isnan(result.signal_based_parameters_.loc["sample_entropy_acc_is"])

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

        assert entropy_single_axis == pytest.approx(entropy_multiple_axes)

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

        pd.testing.assert_series_equal(entropy_from_100_hz, entropy_from_50_hz)

    def test_sufficient_samples_returns_float(self):
        algo = SampleEntropy(acc_columns=["acc_is"], num_samples_threshold=200)
        result = algo.calculate(
            pd.DataFrame(np.random.randn(500, 6), columns=BF_SENSOR_COLS),
            sampling_rate_hz=100,
        )
        df = result.signal_based_parameters_
        assert df.shape == (1,)
        val = df.iloc[0]
        assert isinstance(val, float) and not np.isnan(val) and val > 0
