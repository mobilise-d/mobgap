import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS
from mobgap.secondary_outcomes import (
    StrideParameterCV,
)


class TestMetaStrideParameterCV(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = StrideParameterCV

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS(
            stride_list_columns=["stride_length_m", "cadence_spm", "stride_duration_s"]
        ).calculate(data, stride_list=stride_list)


class TestStrideParameterCV:
    def test_cv_uses_sample_std_ddof_1(self):
        stride_list = pd.DataFrame({"stride_duration_s": [1.0, 1.1, 1.2, 1.3]})
        algo = StrideParameterCV(stride_list_columns=["stride_duration_s"])

        result = algo.calculate(pd.DataFrame(), stride_list=stride_list)
        cv = result.signal_based_parameters_.loc["cv_stride_duration_s"]

        expected = 100 * stride_list["stride_duration_s"].std(ddof=1) / stride_list["stride_duration_s"].mean()
        assert cv == pytest.approx(expected)

        # And explicitly not the ddof=0 value
        wrong = 100 * stride_list["stride_duration_s"].std(ddof=0) / stride_list["stride_duration_s"].mean()
        assert cv != pytest.approx(wrong)

    def test_calculates_coefficient_of_variation_as_percentage(self):
        stride_list = pd.DataFrame({"stride_duration_s": [1.0, 2.0, 3.0]})

        result = StrideParameterCV(stride_list_columns=["stride_duration_s"]).calculate(
            pd.DataFrame(), stride_list=stride_list
        )

        assert result.signal_based_parameters_.loc["cv_stride_duration_s"] == pytest.approx(50.0)

    def test_missing_contact_can_distort_retained_stride_cv(self):
        algo = StrideParameterCV(stride_list_columns=["stride_duration_s"])
        true = algo.clone().calculate(pd.DataFrame(), stride_list=pd.DataFrame({"stride_duration_s": [1.0] * 10}))
        missed = algo.calculate(pd.DataFrame(), stride_list=pd.DataFrame({"stride_duration_s": [1.0] * 8 + [2.0]}))
        assert true.signal_based_parameters_.iloc[0] == 0
        assert missed.signal_based_parameters_.iloc[0] == pytest.approx(30.0)

    def test_no_columns(self):
        algo = StrideParameterCV(stride_list_columns=None)
        result = algo.calculate(
            pd.DataFrame(np.random.randn(1000, 6), columns=BF_SENSOR_COLS), stride_list=pd.DataFrame([0, 101, 1.0])
        )
        assert result is algo
        assert result.signal_based_parameters_.empty

    def test_missing_columns_warns(self):
        stride_list = pd.DataFrame([[0, 91, 0.9], [91, 182, 0.91]], columns=["start", "end", "stride_duration_s"])
        algo = StrideParameterCV(stride_list_columns=["non_existent"])
        with pytest.warns(UserWarning, match="None of .* is available"):
            result = algo.calculate(
                pd.DataFrame(np.random.randn(1000, 6), columns=BF_SENSOR_COLS), stride_list=stride_list
            )
        assert result.signal_based_parameters_.empty

    def test_pipe_data(self):
        algo = StrideParameterCV(stride_list_columns=["stride_length_m", "cadence_spm", "stride_duration_s"])
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
            assert result.signal_based_parameters_[col] > 0
