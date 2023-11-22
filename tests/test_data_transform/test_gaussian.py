import pytest
import numpy as np
from scipy.ndimage import gaussian_filter
from tpcp.testing import TestAlgorithmMixin
from gaitlink.data_transform._gaussian_filter import GaussianFilter

class TestMetaGaussianFilter(TestAlgorithmMixin):
    ALGORITHM_CLASS = GaussianFilter
    __test__ = True

    @pytest.fixture()
    def after_action_instance(self):
        # Creating an instance of your GaussianFilter class with some initial conditions
        gaussian_filter_instance = self.ALGORITHM_CLASS(sigma=1.0)

        # Use some random input data for the meta-test (actual values might not be important)
        input_data = np.random.rand(10, 10, 10)

        # Perform the Gaussian filtering using your GaussianFilter class
        result = gaussian_filter_instance.transform(input_data)

        # Return the GaussianFilter class instance with initial conditions
        return gaussian_filter_instance
