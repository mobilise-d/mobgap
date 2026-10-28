import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS
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
        "values,expected,dtype",
        [
            ([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.log(3), np.float16),
            ([0.0, 0.0, 0.0, 1.0, 0.0], np.inf, np.float64),
            ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], np.nan, np.float64),
            ([2.0] * 6, 0.0, np.float64),
        ],
    )
    def test_exact_common_extendible_template_counts(self, values, expected, dtype):
        # For 000010: m=2 has 3 unordered matching pairs, m=3 has 1.
        # 00010 has one matching pair at m=2 and none at m=3.
        row = (
            SampleEntropy(acc_columns=["acc_is"], r=0.15, num_samples_threshold=0)
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

    @pytest.mark.parametrize("n_samples", [0, 1, 300, 400])
    def test_sample_threshold_is_applied_per_axis_after_resampling(self, n_samples):
        data = pd.DataFrame(
            np.random.default_rng(0).normal(size=(n_samples, 2)),
            columns=["acc_is", "acc_ml"],
        )

        result = SampleEntropy(acc_columns=["acc_is", "acc_ml"], num_samples_threshold=200).calculate(
            data, sampling_rate_hz=100
        )

        pd.testing.assert_series_equal(
            result.signal_based_parameters_,
            pd.Series({"sample_entropy_acc_is": np.nan, "sample_entropy_acc_ml": np.nan}),
        )

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
        entropies = []
        for sampling_rate_hz in [50, 100]:
            time = np.arange(8 * sampling_rate_hz) / sampling_rate_hz
            data = pd.DataFrame({"acc_is": np.sin(2 * np.pi * time) + 0.2 * np.sin(6 * np.pi * time)})
            result = SampleEntropy(acc_columns=["acc_is"]).calculate(data, sampling_rate_hz=sampling_rate_hz)
            entropies.append(result.signal_based_parameters_["sample_entropy_acc_is"])

        assert entropies[0] > 0
        assert entropies[1] == pytest.approx(entropies[0])
