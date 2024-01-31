import numpy as np
import pandas as pd
import pytest
from scipy.signal import savgol_filter
from tpcp.testing import TestAlgorithmMixin
from gaitlink.data_transform import SavgolFilter

class TestSavgolFilter(TestAlgorithmMixin):
    ALGORITHM_CLASS = SavgolFilter
    __test__ = True

    @pytest.fixture()
    def after_action_instance(self):
        # Creating an instance of your SavgolFilter class with some initial conditions
        savgol_filter = self.ALGORITHM_CLASS(window_length=5, polyorder=2)

        # Use some random input data for the meta-test (actual values might not be important)
        input_data = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})

        # Perform the Savgol filtering action using your SavgolFilter class
        savgol_filter.transform(input_data)

        # Return the SavgolFilter class instance with initial conditions
        return savgol_filter

    def test_savgol_filter_transform(self):
        # Create a SavgolFilter instance with window_length and polyorder values
        savgol_filter = self.ALGORITHM_CLASS(window_length=5, polyorder=2)

        # Create a sample DataFrame
        sample_data = pd.DataFrame({"column1": [1.0, 2.0, 3.0], "column2": [4.0, 5.0, 6.0]})

        # Perform the transformation
        transformed_data = savgol_filter.transform(sample_data)

        # Check if 'transformed_data_' is not None
        assert transformed_data.transformed_data_ is not None

        # Check if the transformed_data is not an empty DataFrame
        assert not transformed_data.transformed_data_.empty

        # Calculate the expected output using transform method
        expected_output = savgol_filter.transform(sample_data).transformed_data_

        # Compare the shapes of the transformed data with the expected output
        assert transformed_data.transformed_data_.shape == expected_output.shape

        # Compare the transformed data with the manually filtered data
        pd.testing.assert_frame_equal(transformed_data.transformed_data_, expected_output)


    def test_savgol_filter_empty_data(self):
        # Create a SavgolFilter instance with window_length and polyorder values
        savgol_filter = self.ALGORITHM_CLASS(window_length=5, polyorder=2)

        # Create an empty DataFrame
        data = pd.DataFrame()

        # Attempt to transform with an empty DataFrame (expecting a ValueError)
        with pytest.raises(ValueError, match="Parameter 'data' must be provided."):
            savgol_filter.transform(data)


