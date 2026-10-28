import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

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
    def test_calculates_coefficient_of_variation_as_percentage(self):
        stride_list = pd.DataFrame(
            {"stride_duration_s": [1.0, 2.0, 3.0], "stride_length_m": [2.0, 4.0, 6.0], "cadence_spm": [60.0] * 3}
        )

        result = StrideParameterCV(stride_list_columns=list(stride_list)).calculate(
            pd.DataFrame(), stride_list=stride_list
        )

        # The first two columns have sample SD equal to half the mean; the third is constant.
        pd.testing.assert_series_equal(
            result.signal_based_parameters_,
            pd.Series({"cv_stride_duration_s": 50.0, "cv_stride_length_m": 50.0, "cv_cadence_spm": 0.0}),
        )

    def test_no_columns(self):
        algo = StrideParameterCV(stride_list_columns=None)
        result = algo.calculate(pd.DataFrame(), stride_list=pd.DataFrame())
        assert result is algo
        assert result.signal_based_parameters_.empty

    def test_missing_columns_warns(self):
        stride_list = pd.DataFrame([[0, 91, 0.9], [91, 182, 0.91]], columns=["start", "end", "stride_duration_s"])
        algo = StrideParameterCV(stride_list_columns=["non_existent"])
        with pytest.warns(UserWarning, match="None of .* is available"):
            result = algo.calculate(pd.DataFrame(), stride_list=stride_list)
        assert result.signal_based_parameters_.empty
