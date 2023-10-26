import pandas as pd
import pytest
from gaitlink.data_transform._resample import Resample
from scipy.signal import resample


# Meta Tests
class TestMetaResample:
    @pytest.mark.parametrize("target_sampling_rate_hz", [100.0, 200.0])
    def test_target_sampling_rate(self, target_sampling_rate_hz):
        resampler = Resample(target_sampling_rate_hz)
        assert resampler.target_sampling_rate_hz == target_sampling_rate_hz

# Unit Tests
class TestResample:
    @pytest.mark.parametrize("source_sampling_rate, target_sampling_rate", [(100.0, 100.0), (50.0, 100.0)])
    def test_resample_transform(self, source_sampling_rate, target_sampling_rate):
        # Create a Resample instance with the target sampling rate
        resampler = Resample(target_sampling_rate_hz=target_sampling_rate)

        # Create a sample DataFrame
        sample_data = pd.DataFrame({'acc_x': [1.0, 2.0, 3.0, 4.0], 'acc_y': [0.5, 1.0, 1.5, 2.0]})

        # Perform the transformation with the source sampling rate
        transformed_data = resampler.transform(sample_data, sampling_rate_hz=source_sampling_rate)
        resampling_factor = target_sampling_rate / source_sampling_rate

        # Calculate the expected output using scipy.signal.resample
        expected_output = pd.DataFrame(resample(sample_data, int(len(sample_data) * resampling_factor)))
        expected_output.columns = ['acc_x', 'acc_y']

        pd.testing.assert_frame_equal(transformed_data.transformed_data_, expected_output)

    @pytest.mark.parametrize("source_sampling_rate, target_sampling_rate", [(100.0, 100.0), (50.0, 100.0)])
    def test_resample_attribute_update(self, source_sampling_rate, target_sampling_rate):
        # Create a Resample instance with the target sampling rate
        resampler = Resample(target_sampling_rate_hz=target_sampling_rate)  # Provide the target_sampling_rate

        # Create a sample DataFrame
        sample_data = pd.DataFrame({'acc_x': [1.0, 2.0, 3.0, 4.0], 'acc_y': [0.5, 1.0, 1.5, 2.0]})

        # Perform the transformation with the source sampling rate
        transformed_data = resampler.transform(sample_data, sampling_rate_hz=source_sampling_rate)

        # Check if the 'transformed_data_' attribute is updated with the transformed data
        expected_resampled_data = pd.DataFrame(
            data=resample(sample_data, int(len(sample_data) * (target_sampling_rate / source_sampling_rate))),
            columns=sample_data.columns
        )

        # Compare the transformed data with the manually resampled data
        pd.testing.assert_frame_equal(transformed_data.transformed_data_, expected_resampled_data)

# Regression Test
class TestResampleRegression:
    @pytest.mark.parametrize("target_sampling_rate", [100.0, 200.0])
    def test_regression_test(self, target_sampling_rate):
        # Load your real data or create a synthetic dataset
        real_data = pd.DataFrame({'acc_x': [1.0, 2.0, 3.0, 4.0], 'acc_y': [0.5, 1.0, 1.5, 2.0]})

        # Create a Resample instance with the target sampling rate
        resampler = Resample(target_sampling_rate_hz=target_sampling_rate)

        # Transform the real data
        transformed_data = resampler.transform(real_data, sampling_rate_hz=50.0)

        # Calculate the expected output using scipy.signal.resample
        expected_output = pd.DataFrame(resample(real_data, int(len(real_data) * (target_sampling_rate / 50.0)), axis=0))
        expected_output.columns = ['acc_x', 'acc_y']
        # Check if the transformed data matches the expected output
        pd.testing.assert_frame_equal(transformed_data.transformed_data_, expected_output)


# Run the tests
if __name__ == '__main__':
    pytest.main()
