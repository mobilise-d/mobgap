from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scipy.signal import filtfilt, lfilter
from tpcp.testing import TestAlgorithmMixin

from gaitlink.data_transform import (
    ButterworthFilter,
    EpflDedriftedGaitFilter,
    EpflDedriftFilter,
    EpflGaitFilter,
    FirFilter,
)
from gaitlink.data_transform.base import FixedFilter


class TestMetaEpflGaitFilter(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = EpflGaitFilter
    _IGNORED_NAMES = ["EXPECTED_SAMPLING_RATE_HZ"]

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().filter(pd.DataFrame(np.zeros((500, 3))), sampling_rate_hz=40.0)


class TestMetaButterworthFilter(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = ButterworthFilter
    ONLY_DEFAULT_PARAMS = False

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS(2, 30).filter(pd.DataFrame(np.zeros((500, 3))), sampling_rate_hz=100.0)


class TestMetaFirFilter(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = FirFilter
    ONLY_DEFAULT_PARAMS = False

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS(2, 30).filter(pd.DataFrame(np.zeros((500, 3))), sampling_rate_hz=100.0)


class TestMetaEpflDedriftFilter(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = EpflDedriftFilter
    _IGNORED_NAMES = ["EXPECTED_SAMPLING_RATE_HZ"]

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().filter(pd.DataFrame(np.zeros((500, 3))), sampling_rate_hz=40.0)


class TestMetaEpflDedriftedGaitFilter(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = EpflDedriftedGaitFilter

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().filter(pd.DataFrame(np.zeros((500, 3))), sampling_rate_hz=40.0)


class TestFixedFilter:
    filter_subclass: type[FixedFilter]

    @pytest.fixture(autouse=True, params=(EpflDedriftFilter, EpflGaitFilter))
    def get_algorithm(self, request):
        self.filter_subclass = request.param

    @pytest.mark.parametrize("zero_phase", [True, False])
    @pytest.mark.parametrize("dtype", [np.array, pd.DataFrame, pd.Series, "array_1d"])
    def test_simple_filter(self, zero_phase, dtype):
        data_raw = np.random.rand(1000, 1)
        if dtype == pd.DataFrame:
            data = pd.DataFrame(data_raw)
        elif dtype == pd.Series:
            data = pd.Series(data_raw[:, 0])
        elif dtype == "array_1d":
            data = data_raw[:, 0]
        else:
            data = data_raw

        result = self.filter_subclass(zero_phase=zero_phase).filter(data, sampling_rate_hz=40.0)

        if zero_phase:
            reference = filtfilt(*self.filter_subclass().coefficients, data_raw, axis=0)
        else:
            reference = lfilter(*self.filter_subclass().coefficients, data_raw, axis=0)

        if dtype in (pd.Series, "array_1d"):
            reference = pd.Series(reference[:, 0])
        assert np.allclose(result.transformed_data_, reference)
        assert type(result.transformed_data_) == type(data)

        if dtype == pd.DataFrame:
            assert result.transformed_data_.columns == data.columns

        if dtype in (pd.Series, pd.DataFrame):
            assert result.transformed_data_.index.equals(data.index)

        assert result.transformed_data_ is result.filtered_data_

    def test_error_on_wrong_sampling_rate(self):
        with pytest.raises(ValueError):
            self.filter_subclass().filter(
                pd.DataFrame(np.zeros((500, 3))), sampling_rate_hz=self.filter_subclass.EXPECTED_SAMPLING_RATE_HZ + 1
            )

    def test_error_no_sampling_rate(self):
        with pytest.raises(ValueError):
            self.filter_subclass().filter(pd.DataFrame(np.zeros((500, 3))))


class TestEpflGaitFilter:
    def test_matlab_comparison(self):
        """This actually tests if the output is identical to the output of the matlab code."""
        HERE = Path(__file__).parent
        data_folder = HERE / "epfl_filter_example_data"

        input_data = pd.read_csv(data_folder / "example_input.csv")
        expected_output = pd.read_csv(data_folder / "mat_output.csv", header=None)
        expected_output.columns = ["x", "y", "z"]

        result = EpflGaitFilter(zero_phase=True).filter(input_data, sampling_rate_hz=40.0)

        assert_frame_equal(result.transformed_data_, expected_output, atol=1e-5)


class TestEpflDedriftedGaitFilter:
    def test_equivalence_to_manual(self):
        HERE = Path(__file__).parent
        data_folder = HERE / "epfl_filter_example_data"

        input_data = pd.read_csv(data_folder / "example_input.csv")

        direct_output = EpflDedriftedGaitFilter().filter(input_data, sampling_rate_hz=40.0).filtered_data_

        cascaded_output = (
            EpflGaitFilter()
            .filter(EpflDedriftFilter().filter(input_data, sampling_rate_hz=40.0).filtered_data_, sampling_rate_hz=40.0)
            .filtered_data_
        )

        assert_frame_equal(direct_output, cascaded_output, atol=1e-5)
