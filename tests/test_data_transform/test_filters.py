from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal
from scipy.signal import butter, filtfilt, firwin, lfilter, sosfilt, sosfiltfilt
from tpcp.testing import TestAlgorithmMixin

from mobgap.data import LabExampleDataset
from mobgap.data_transform import (
    ButterworthFilter,
    EpflDedriftedGaitFilter,
    EpflDedriftFilter,
    EpflGaitFilter,
    FirFilter,
    HampelFilter,
)
from mobgap.data_transform.base import FixedFilter

HERE = Path(__file__).parent


@pytest.fixture(params=[np.array, pd.DataFrame, pd.Series, "array_1d"])
def supported_dtypes(request):
    dtype = request.param

    def conversion_func(data_raw):
        assert isinstance(data_raw, pd.DataFrame)
        if dtype == pd.DataFrame:
            data = data_raw
        elif dtype == pd.Series:
            data = data_raw.iloc[:, 0]
        elif dtype == "array_1d":
            data = data_raw.to_numpy()[:, 0]
        else:
            data = data_raw.to_numpy()
        return data

    def assertions(result, reference, input_data):
        if dtype in (pd.Series, "array_1d"):
            reference = pd.Series(reference)
        assert np.allclose(result, reference)

        assert type(result) == type(input_data)

        if dtype == pd.DataFrame:
            assert set(result.columns) == set(input_data.columns)

        if dtype in (pd.Series, pd.DataFrame):
            assert result.index.equals(input_data.index)

    return conversion_func, assertions


class TestMetaEpflGaitFilter(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = EpflGaitFilter
    _IGNORED_NAMES = ["EXPECTED_SAMPLING_RATE_HZ"]

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().filter(pd.DataFrame(np.zeros((500, 3))), sampling_rate_hz=40.0)


class TestMetaButterworthFilter(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = ButterworthFilter
    ONLY_DEFAULT_PARAMS = False

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS(2, 30).filter(pd.DataFrame(np.zeros((500, 3))), sampling_rate_hz=100.0)


class TestMetaFirFilter(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = FirFilter
    ONLY_DEFAULT_PARAMS = False

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS(2, 30).filter(pd.DataFrame(np.zeros((500, 3))), sampling_rate_hz=100.0)


class TestMetaEpflDedriftFilter(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = EpflDedriftFilter
    _IGNORED_NAMES = ["EXPECTED_SAMPLING_RATE_HZ"]

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().filter(pd.DataFrame(np.zeros((500, 3))), sampling_rate_hz=40.0)


class TestMetaEpflDedriftedGaitFilter(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = EpflDedriftedGaitFilter

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().filter(pd.DataFrame(np.zeros((500, 3))), sampling_rate_hz=40.0)


class TestMetaHampelFilter(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = HampelFilter
    ONLY_DEFAULT_PARAMS = False

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS(2, 30).filter(pd.DataFrame(np.zeros((500, 1))), sampling_rate_hz=100.0)


class TestFixedFilter:
    filter_subclass: type[FixedFilter]

    @pytest.fixture(autouse=True, params=(EpflDedriftFilter, EpflGaitFilter))
    def get_algorithm(self, request):
        self.filter_subclass = request.param

    @pytest.mark.parametrize("zero_phase", [True, False])
    def test_simple_filter(self, zero_phase, supported_dtypes):
        conversion_func, output_assertions = supported_dtypes

        data_raw = pd.DataFrame(np.random.rand(1000, 2))
        data_raw.index += 2

        data = conversion_func(data_raw)
        result = self.filter_subclass(zero_phase=zero_phase).filter(data, sampling_rate_hz=40.0)

        if zero_phase:
            reference = filtfilt(*self.filter_subclass().coefficients, data, axis=0)
        else:
            reference = lfilter(*self.filter_subclass().coefficients, data, axis=0)

        assert result.transformed_data_ is result.filtered_data_

        output_assertions(result.transformed_data_, reference, data)

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


class TestButterworthFilter:
    @pytest.mark.parametrize("zero_phase", (True, False))
    @pytest.mark.parametrize("order", (2, 30))
    def test_equivalence_to_manual(self, zero_phase, order, supported_dtypes):
        conversion_func, output_assertions = supported_dtypes

        cutoff = 10
        sampling_rate = 100
        f = ButterworthFilter(zero_phase=zero_phase, order=order, cutoff_freq_hz=cutoff)

        raw_data = LabExampleDataset()[0].data_ss[["gyr_x", "gyr_y"]]
        data = conversion_func(raw_data)
        result = f.filter(data, sampling_rate_hz=100).filtered_data_

        if zero_phase:
            reference = sosfiltfilt(butter(order, cutoff, fs=sampling_rate, output="sos"), data, axis=0)
        else:
            reference = sosfilt(butter(order, cutoff, fs=sampling_rate, output="sos"), data, axis=0)

        output_assertions(result, reference, data)


class TestFirFilter:
    @pytest.mark.parametrize("zero_phase", (True, False))
    @pytest.mark.parametrize("window", ("hamming", ("kaiser", 2)))
    def test_equivalence_to_manual(self, zero_phase, window, supported_dtypes):
        conversion_func, output_assertions = supported_dtypes

        cutoff = 10
        order = 2
        sampling_rate = 100
        f = FirFilter(zero_phase=zero_phase, order=order, cutoff_freq_hz=cutoff, window=window)

        raw_data = LabExampleDataset()[0].data_ss[["gyr_x", "gyr_y"]]
        data = conversion_func(raw_data)
        result = f.filter(data, sampling_rate_hz=100).filtered_data_

        b = firwin(order + 1, cutoff, fs=sampling_rate, pass_zero="lowpass", window=window)

        reference = filtfilt(b, 1, data, axis=0) if zero_phase else lfilter(b, 1, data, axis=0)

        output_assertions(result, reference, data)


class TestHampelFilter:
    def test_matlab_equivalent(self):
        mat_data = scipy.io.loadmat(HERE / "hampel_filter_test_data/matlab_hampel_test_data.mat")
        data = mat_data["data"].flatten()
        matlab_filtered_array = mat_data["filteredDataArray"]
        window_sizes = mat_data["windowSizes"].flatten()
        num_stds = mat_data["numStds"].flatten()

        # Iterate through the parameter combinations
        for i in range(len(window_sizes)):
            window_size = int(window_sizes[i])
            num_std = num_stds[i]

            python_filtered = HampelFilter(window_size, num_std).filter(data, sampling_rate_hz=100.0).filtered_data_
            # MATLAB output
            matlab_filtered = matlab_filtered_array[i, 0].flatten()

            assert_array_almost_equal(python_filtered, matlab_filtered)
