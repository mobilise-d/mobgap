import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS
from mobgap.secondary_outcomes import (
    RootMeanSquare,
)


class TestMetaRootMeanSquare(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = RootMeanSquare

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS().calculate(data)


class TestRootMeanSquare:
    def test_no_columns_warns_and_returns_empty(self):
        data = pd.DataFrame({"temperature": [20.0, 22.0]})

        with pytest.warns(UserWarning, match="No acceleration or gyroscope columns"):
            result = RootMeanSquare().calculate(data)

        assert result.signal_based_parameters_.empty

    def test_gyroscope_rms_is_available_without_acceleration(self):
        data = pd.DataFrame({"gyr_is": [3.0, 3.0], "gyr_ml": [-4.0, -4.0]})

        result = RootMeanSquare(gyr_columns=["gyr_is", "gyr_ml"]).calculate(data).signal_based_parameters_

        expected = pd.Series({"rms_gyr_is": 3.0, "rms_gyr_ml": 4.0})
        pd.testing.assert_series_equal(result, expected)

    @pytest.mark.parametrize("level", [1.0, 0.1])
    def test_constant_signal(self, level):
        algo = RootMeanSquare(acc_columns=["acc_is", "acc_ml", "acc_pa"], gyr_columns=["gyr_is", "gyr_ml", "gyr_pa"])
        result = algo.calculate(pd.DataFrame(np.full((100, 6), level), columns=BF_SENSOR_COLS))
        df = result.signal_based_parameters_

        for col in ["rms_acc_is", "rms_acc_ml", "rms_acc_pa"]:
            assert df[col] == pytest.approx(0.0, abs=1e-12)
        for col in ["rms_gyr_is", "rms_gyr_ml", "rms_gyr_pa"]:
            assert df[col] == pytest.approx(level, abs=1e-12)
        assert df["rms_total_acc"] == 0
        assert df.filter(like="rms_ratio_").isna().all().all()

    def test_integer_inputs_are_promoted_before_centering_and_squaring(self):
        data = pd.DataFrame(
            {"acc_is": np.array([-100, 100], dtype=np.int8), "gyr_is": np.array([100, 100], dtype=np.int8)}
        )
        row = RootMeanSquare(acc_columns=["acc_is"], gyr_columns=["gyr_is"]).calculate(data).signal_based_parameters_
        assert row.rms_acc_is == 100
        assert row.rms_gyr_is == 100

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

        result = (
            RootMeanSquare(acc_columns=["acc_is", "acc_ml", "acc_pa"], gyr_columns=["gyr_is", "gyr_ml", "gyr_pa"])
            .calculate(data)
            .signal_based_parameters_
        )

        expected_total = np.sqrt(5)
        assert result["rms_total_acc"] == pytest.approx(expected_total)
        assert result["rms_ratio_acc_is"] == pytest.approx(1 / expected_total)
        assert result["rms_ratio_acc_ml"] == pytest.approx(2 / expected_total)
        assert result["rms_ratio_acc_pa"] == pytest.approx(0)

    def test_acceleration_rms_removes_only_dc_offset(self):
        data = pd.DataFrame({"acc_is": [10.0, 11.0, 12.0]})

        result = RootMeanSquare(acc_columns=["acc_is"]).calculate(data).signal_based_parameters_

        expected_rms = np.sqrt(2 / 3)
        assert result["rms_acc_is"] == pytest.approx(expected_rms)
        assert np.isnan(result["rms_total_acc"])
        assert np.isnan(result["rms_ratio_acc_is"])

    def test_missing_anatomical_axis_does_not_publish_partial_total(self):
        data = pd.DataFrame({"acc_is": [1.0, -1.0], "acc_ml": [2.0, -2.0]})
        row = RootMeanSquare(acc_columns=["acc_is", "acc_ml", "acc_pa"]).calculate(data).signal_based_parameters_
        assert row["rms_acc_is"] == 1
        assert row["rms_acc_ml"] == 2
        assert np.isnan(row["rms_total_acc"])
        assert np.isnan(row["rms_ratio_acc_is"])
