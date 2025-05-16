import numpy as np
import pandas as pd
import pytest
from scipy.signal import savgol_filter
from tpcp.testing import TestAlgorithmMixin

from mobgap.data_transform import SavgolFilter


class TestMetaSavgolFilter(TestAlgorithmMixin):
    ALGORITHM_CLASS = SavgolFilter
    ONLY_DEFAULT_PARAMS = False
    __test__ = True

    @pytest.fixture
    def after_action_instance(self):
        # Creating an instance of your SavgolFilter class with some initial conditions
        savgol_filter = self.ALGORITHM_CLASS(window_length_s=5, polyorder_rel=0.2)

        # Use some random input data for the meta-test (actual values might not be important)
        input_data = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})

        # Perform the Savgol filtering action using your SavgolFilter class
        savgol_filter.transform(input_data, sampling_rate_hz=100)

        # Return the SavgolFilter class instance with initial conditions
        return savgol_filter


class TestSavgolFilter:
    def test_savgol_filter_transform(self):
        # Create a SavgolFilter instance with window_length and polyorder values

        sampling_rate = 10
        window_length = 5
        polyorder = 2

        sample_data = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        # Perform the transformation
        savgolfilter = SavgolFilter(
            window_length_s=window_length / sampling_rate, polyorder_rel=polyorder / window_length
        )
        transformed_data = savgolfilter.filter(sample_data, sampling_rate_hz=sampling_rate).filtered_data_

        # Calculate the expected output using transform method
        expected_output = savgol_filter(sample_data, window_length=window_length, polyorder=polyorder, mode="mirror")

        # Compare the transformed data with the manually filtered data
        np.testing.assert_array_equal(transformed_data, expected_output)
