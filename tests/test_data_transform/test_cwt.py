import numpy as np
import pytest
from pywt import cwt, scale2frequency
from tpcp.testing import TestAlgorithmMixin

from mobgap.data_transform import CwtFilter


class TestMetaCwtFilter(TestAlgorithmMixin):
    ALGORITHM_CLASS = CwtFilter
    __test__ = True
    ONLY_DEFAULT_PARAMS = False

    @pytest.fixture
    def after_action_instance(self):
        # Creating an instance of your CwtFilter class with some initial conditions
        cwt_filter_instance = self.ALGORITHM_CLASS()

        # Use some random input data for the meta-test (actual values might not be important)
        input_data = np.array([1.0, 2.0, 3.0, 4.0])
        # Perform the Continuous Wavelet Transform using your CwtFilter class
        result = cwt_filter_instance.transform(input_data, sampling_rate_hz=100)

        # Return the CwtFilter class instance with initial conditions
        return result


class TestCwtFilter:
    def test_compare_against_pywt(self):
        wavelet = "gaus2"
        scale = 10
        equivalent_freq = scale2frequency(wavelet, scale) * 100

        cwt_filter = CwtFilter(wavelet=wavelet, center_frequency_hz=equivalent_freq)

        data = np.random.random((1000, 2))

        cwt_filter.filter(data, sampling_rate_hz=100)

        # Compare to calling cwt directly
        first_col, _ = cwt(data[:, 0], [scale], wavelet)
        second_col, _ = cwt(data[:, 1], [scale], wavelet)

        np.testing.assert_equal(cwt_filter.transformed_data_[:, 0], first_col.flatten())
        np.testing.assert_equal(cwt_filter.transformed_data_[:, 1], second_col.flatten())

        assert cwt_filter.transformed_data_.shape == data.shape
