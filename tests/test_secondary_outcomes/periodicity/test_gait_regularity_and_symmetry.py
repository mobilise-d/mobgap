import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS
from mobgap.secondary_outcomes import (
    GaitRegularityAndSymmetry,
)


class TestMetaGaitRegularityAndSymmetry(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = GaitRegularityAndSymmetry

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS().calculate(data, sampling_rate_hz=sampling_rate_hz, replicate_matlab=True)


class TestGaitRegularityAndSymmetry:
    def test_missing_columns_returns_self(self):
        data_missing = pd.DataFrame({"acc_is": np.random.randn(100)})
        algo = GaitRegularityAndSymmetry()
        result = algo.calculate(data_missing, sampling_rate_hz=100, replicate_matlab=False)
        assert result is algo
        assert result.signal_based_parameters_.empty

    def test_pipe_data(self):
        algo = GaitRegularityAndSymmetry()
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
        assert set(df.index) == set(expected)

        row = df

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
