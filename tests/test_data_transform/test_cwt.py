import numpy as np
import pytest
from scipy.signal import cwt, morlet, ricker
from tpcp.testing import TestAlgorithmMixin

from gaitlink.data_transform._cwt import CwtFilter


class TestMetaCwtFilter(TestAlgorithmMixin):
    ALGORITHM_CLASS = CwtFilter
    __test__ = True
    ONLY_DEFAULT_PARAMS = False

    @pytest.fixture()
    def after_action_instance(self):
        # Creating an instance of your CwtFilter class with some initial conditions
        cwt_filter_instance = self.ALGORITHM_CLASS()

        # Use some random input data for the meta-test (actual values might not be important)
        input_data = np.array([1.0, 2.0, 3.0, 4.0])
        # Perform the Continuous Wavelet Transform using your CwtFilter class
        result = cwt_filter_instance.transform(input_data)

        # Return the CwtFilter class instance with initial conditions
        return result


class TestCwtFilter:
    def test_compare_against_scipy(self):
        wavelet = ricker
        width = 10

        cwt_filter = CwtFilter(wavelet=wavelet, width=width)

        data = np.random.random((1000, 2))

        cwt_filter.transform(data)

        # Compare to calling cwt directly
        first_col = cwt(data[:, 0], wavelet, [width])
        second_col = cwt(data[:, 1], wavelet, [width])

        np.testing.assert_equal(cwt_filter.transformed_data_[:, 0], first_col.flatten())
        np.testing.assert_equal(cwt_filter.transformed_data_[:, 1], second_col.flatten())

        assert cwt_filter.transformed_data_.shape == data.shape


