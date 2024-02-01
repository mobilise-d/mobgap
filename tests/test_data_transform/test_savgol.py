import numpy as np
import pandas as pd
import pytest
from scipy.signal import savgol_filter
from tpcp.testing import TestAlgorithmMixin

from gaitlink.data_transform import SavgolFilter


class TestMetaSavgolFilter(TestAlgorithmMixin):
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


class TestSavgolFilter:
    def test_savgol_filter_transform(self):
        # Create a SavgolFilter instance with window_length and polyorder values
        savgolfilter = SavgolFilter()

        # Create a sample DataFrame
        # sample_data = pd.DataFrame({"column1": [1.0, 2.0, 3.0], "column2": [4.0, 5.0, 6.0]})

        sample_data = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        # Perform the transformation
        transformed_data = savgolfilter.transform(sample_data)

        # Calculate the expected output using transform method
        expected_output = savgol_filter(sample_data, window_length=5, polyorder=2, mode="mirror")

        # Converting the data to Pandas Dataframe for easy comparison
        expected_output_df = pd.DataFrame(expected_output)

        # Convert the transformed data to a DataFrame
        transformed_data_df = pd.DataFrame(transformed_data.transformed_data_)

        # Compare the shapes of the transformed data with the expected output
        assert tuple(transformed_data_df.shape) == expected_output_df.shape

        # Compare the transformed data with the manually filtered data
        pd.testing.assert_frame_equal(transformed_data_df, expected_output_df)
