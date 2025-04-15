import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter1d
from tpcp.testing import TestAlgorithmMixin

from mobgap.data_transform import GaussianFilter


class TestMetaGaussianFilter(TestAlgorithmMixin):
    ALGORITHM_CLASS = GaussianFilter
    __test__ = True

    @pytest.fixture
    def after_action_instance(self):
        # Creating an instance of your GaussianFilter class with some initial conditions
        gaussian_filter = self.ALGORITHM_CLASS()

        # Use some random input data for the meta-test (actual values might not be important)
        input_data = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})

        # Perform the Gaussian filtering action using your GaussianFilter class
        gaussian_filter.transform(input_data, sampling_rate_hz=100)

        # Return the GaussianFilter class instance with initial conditions
        return gaussian_filter


class TestGaussianFilter:
    def test_gaussian_filter_transform(self):
        # Create a GaussianFilter instance with a sigma value
        gaussian_filter = GaussianFilter(sigma_s=1.0)

        # Create a sample DataFrame
        sample_data = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        # Perform the transformation
        transformed_data = gaussian_filter.transform(sample_data, sampling_rate_hz=100)

        # Calculate the expected output using scipy.ndimage.gaussian_filter1d
        expected_output = pd.DataFrame(gaussian_filter1d(sample_data, sigma=1.0 * 100, axis=0))

        # Compare the shapes of the transformed data with the expected output
        assert transformed_data.transformed_data_.shape == expected_output.shape

        # Compare the transformed data with the manually filtered data
        np.testing.assert_array_equal(transformed_data.transformed_data_, expected_output.to_numpy())

    # Add more test cases as needed
