import pandas as pd
import pytest
from gaitlink.data_transform._resample import Resample
from scipy.signal import resample
from tpcp.testing import TestAlgorithmMixin
from gaitlink.data import LabExampleDataset

class TestMetaResample(TestAlgorithmMixin):
    ALGORITHM_CLASS = Resample
    __test__ = True

    @pytest.fixture
    def after_action_instance(self):
        # Creating an instance of your Resample class with some initial conditions
        resampler = self.ALGORITHM_CLASS(target_sampling_rate_hz=100.0)

        # Use some random input data for the meta-test (actual values might not be important)
        input_data = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})
        current_sampling_rate = 50.0  # For example purposes

        # Perform the resampling action using your Resample class
        result = resampler.transform(input_data, sampling_rate_hz=current_sampling_rate)

        # Return the Resample class instance with initial conditions
        return resampler

# Unit Tests
class TestResample:
    @pytest.mark.parametrize("source_sampling_rate, target_sampling_rate", [(100.0, 100.0), (50.0, 100.0), (100.0, 50.0)])
    def test_resample_transform(self, source_sampling_rate, target_sampling_rate):
        # Create a Resample instance with the target sampling rate
        resampler = Resample(target_sampling_rate_hz=target_sampling_rate)

        # Create a sample DataFrame
        sample_data = pd.DataFrame({'acc_x': [1.0, 2.0, 3.0, 4.0], 'acc_y': [0.5, 1.0, 1.5, 2.0]})

        # Perform the transformation with the source sampling rate
        transformed_data = resampler.transform(sample_data, sampling_rate_hz=source_sampling_rate)

        # Calculate the expected output using scipy.signal.resample
        resampling_factor = target_sampling_rate / source_sampling_rate
        expected_output = pd.DataFrame(resample(sample_data, int(len(sample_data) * resampling_factor)))
        expected_output.columns = ['acc_x', 'acc_y']

        # Check if the 'transformed_data_' attribute is updated with the transformed data
        expected_resampled_data = pd.DataFrame(
            data=resample(sample_data, int(len(sample_data) * resampling_factor)),
            columns=sample_data.columns
        )

        # Compare the transformed data with the manually resampled data
        pd.testing.assert_frame_equal(transformed_data.transformed_data_, expected_output)

        # Check if the transformed data is a copy when source and target sampling rates are identical
        if source_sampling_rate == target_sampling_rate:
            assert transformed_data is not sample_data  # Check that transformed_data is not the same object as sample_data



example_data = LabExampleDataset()
ha_example_data = example_data.get_subset(cohort="HA")
single_test = ha_example_data.get_subset(participant_id="002", test="Test11", trial="Trial1")
df = single_test.data["LowerBack"]
# Regression Test
class TestResampleRegression:
    @pytest.mark.parametrize("target_sampling_rate", [100.0, 200.0])
    def test_regression_test(self, target_sampling_rate):
        # Load your real data or create a synthetic dataset
        real_data = df

        # Create a Resample instance with the target sampling rate
        resampler = Resample(target_sampling_rate_hz=target_sampling_rate)

        # Transform the real data
        transformed_data = resampler.transform(real_data, sampling_rate_hz=50.0)

        # Calculate the expected output using scipy.signal.resample
        expected_output = pd.DataFrame(resample(real_data, int(len(real_data) * (target_sampling_rate / 50.0)), axis=0))
        expected_output.columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
        # Check if the transformed data matches the expected output
        pd.testing.assert_frame_equal(transformed_data.transformed_data_, expected_output)


# Run the tests
if __name__ == '__main__':
    pytest.main()
