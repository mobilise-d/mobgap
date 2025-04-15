import pandas as pd
import pytest
from scipy.signal import resample
from tpcp.testing import TestAlgorithmMixin

from mobgap.data import LabExampleDataset
from mobgap.data_transform import Resample


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
        resampler.transform(input_data, sampling_rate_hz=current_sampling_rate)

        # Return the Resample class instance with initial conditions
        return resampler


class TestResample:
    @pytest.mark.parametrize(
        ("source_sampling_rate", "target_sampling_rate"), [(100.0, 100.0), (50.0, 100.0), (100.0, 50.0)]
    )
    @pytest.mark.parametrize("attempt_index_resample", [True, False])
    def test_resample_transform(self, source_sampling_rate, target_sampling_rate, attempt_index_resample):
        # Create a Resample instance with the target sampling rate
        resampler = Resample(
            target_sampling_rate_hz=target_sampling_rate, attempt_index_resample=attempt_index_resample
        )

        # Create a sample DataFrame
        sample_data = pd.DataFrame({"acc_x": [1.0, 2.0, 3.0, 4.0], "acc_y": [0.5, 1.0, 1.5, 2.0]})

        # Perform the transformation with the source sampling rate
        transformed_data = resampler.transform(sample_data, sampling_rate_hz=source_sampling_rate)

        # Check if 'transformed_data_' is not None
        assert transformed_data.transformed_data_ is not None

        # Check if the transformed_data is not an empty DataFrame
        assert not transformed_data.transformed_data_.empty

        # Calculate the expected output using scipy.signal.resample
        resampling_factor = target_sampling_rate / source_sampling_rate
        if attempt_index_resample:
            data, index = resample(sample_data, int(len(sample_data) * resampling_factor), t=sample_data.index)
        else:
            data = resample(sample_data, int(len(sample_data) * resampling_factor))
            index = None
        expected_output = pd.DataFrame(data, index=index)
        expected_output.columns = ["acc_x", "acc_y"]

        # Compare the shapes of the transformed data with the expected output
        assert transformed_data.transformed_data_.shape == expected_output.shape

        # Compare the transformed data with the manually resampled data
        pd.testing.assert_frame_equal(transformed_data.transformed_data_, expected_output)

        # Check if the transformed data is not the same object when source and target sampling rates are identical
        if source_sampling_rate == target_sampling_rate:
            assert transformed_data is not sample_data

    def test_warn_non_numeric_index(self):
        input_data = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]}, index=["a", "b", "c"])

        resampler = Resample(target_sampling_rate_hz=100.0)

        with pytest.warns(UserWarning):
            resampler.transform(input_data, sampling_rate_hz=50.0)

        assert resampler.transformed_data_.index.tolist() == [0, 1, 2, 3, 4, 5]

    def test_error_if_no_sampling_rate(self):
        input_data = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})

        resampler = Resample(target_sampling_rate_hz=100.0)

        with pytest.raises(ValueError):
            resampler.transform(input_data)

    @pytest.mark.parametrize("target_sampling_rate", [100.0, 200.0])
    def test_regression_test(self, target_sampling_rate):
        example_data = LabExampleDataset()
        ha_example_data = example_data.get_subset(cohort="HA")
        single_test = ha_example_data.get_subset(participant_id="002", test="Test11", trial="Trial1")
        df = single_test.data_ss
        # Load your real data or create a synthetic dataset
        real_data = df

        # Create a Resample instance with the target sampling rate
        resampler = Resample(target_sampling_rate_hz=target_sampling_rate, attempt_index_resample=False)

        # Transform the real data

        transformed_data = resampler.transform(real_data, sampling_rate_hz=50.0)

        # Calculate the expected output using scipy.signal.resample
        expected_output = pd.DataFrame(resample(real_data, int(len(real_data) * (target_sampling_rate / 50.0)), axis=0))
        expected_output.columns = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
        # Check if the transformed data matches the expected output
        pd.testing.assert_frame_equal(transformed_data.transformed_data_, expected_output)
