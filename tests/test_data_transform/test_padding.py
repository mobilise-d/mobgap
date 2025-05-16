import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.data_transform import Crop, Pad


class TestMetaPad(TestAlgorithmMixin):
    ALGORITHM_CLASS = Pad
    ONLY_DEFAULT_PARAMS = False
    __test__ = True

    @pytest.fixture
    def after_action_instance(self):
        # Creating an instance of your Pad class with some initial conditions
        pad = self.ALGORITHM_CLASS(pad_len_s=1.0)

        # Use some random input data for the meta-test (actual values might not be important)
        input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Perform the padding action using your Pad class
        pad.transform(input_data, sampling_rate_hz=100)

        # Return the Pad class instance with initial conditions
        return pad


class TestPad:
    def test_simple_case_constant(self):
        pad = Pad(pad_len_s=1.0, mode="constant", constant_values=0)

        input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        transformed_data = pad.transform(input_data, sampling_rate_hz=1).transformed_data_

        # Padding is only expect to be happening in the "row" direction
        expected_output = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]])

        assert transformed_data.shape == expected_output.shape
        np.testing.assert_array_equal(transformed_data, expected_output)

    def test_unequal_padding(self):
        pad = Pad(pad_len_s=(1.0, 2.0), mode="constant", constant_values=0)

        input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        transformed_data = pad.transform(input_data, sampling_rate_hz=1).transformed_data_

        # Padding is only expect to be happening in the "row" direction
        expected_output = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0], [0, 0, 0]])

        assert transformed_data.shape == expected_output.shape

    def test_default_reflect(self):
        pad = Pad(pad_len_s=1.0)

        input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        transformed_data = pad.transform(input_data, sampling_rate_hz=1).transformed_data_

        # Padding is only expect to be happening in the "row" direction
        expected_output = np.array([[4, 5, 6], [1, 2, 3], [4, 5, 6], [7, 8, 9], [4, 5, 6]])

        assert transformed_data.shape == expected_output.shape
        np.testing.assert_array_equal(transformed_data, expected_output)

    def test_warn_if_index(self):
        pad = Pad(pad_len_s=1.0)

        input_data = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.warns(UserWarning):
            pad.transform(input_data, sampling_rate_hz=1)

    def test_invalid_tuple_length(self):
        pad = Pad(pad_len_s=(1.0, 2.0, 3.0))

        input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(ValueError):
            pad.transform(input_data, sampling_rate_hz=1)

    def test_no_sampling_rate(self):
        pad = Pad(pad_len_s=1.0)

        input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(ValueError):
            pad.transform(input_data, sampling_rate_hz=None)

    def test_roundtrip_converted(self):
        pad = Pad(pad_len_s=(1.0, 2.0), mode="constant", constant_values=0)

        input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        transformed_data = pad.transform(input_data, sampling_rate_hz=1).transformed_data_

        final_data = pad.get_inverse_transformer().transform(transformed_data, sampling_rate_hz=1).transformed_data_

        assert final_data.shape == input_data.shape
        np.testing.assert_array_equal(final_data, input_data)


class TestMetaCrop(TestAlgorithmMixin):
    ALGORITHM_CLASS = Crop
    ONLY_DEFAULT_PARAMS = False
    __test__ = True

    @pytest.fixture
    def after_action_instance(self):
        # Creating an instance of your Crop class with some initial conditions
        crop = self.ALGORITHM_CLASS(crop_len_s=1.0)

        # Use some random input data for the meta-test (actual values might not be important)
        input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Perform the cropping action using your Crop class
        crop.transform(input_data, sampling_rate_hz=1)

        # Return the Crop class instance with initial conditions
        return crop


class TestCrop:
    def test_simple_case(self):
        crop = Crop(crop_len_s=1.0)

        input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        transformed_data = crop.transform(input_data, sampling_rate_hz=1).transformed_data_

        expected_output = np.array([[4, 5, 6]])

        assert transformed_data.shape == expected_output.shape
        np.testing.assert_array_equal(transformed_data, expected_output)

    def test_unequal_crop(self):
        crop = Crop(crop_len_s=(1.0, 2.0))

        input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        transformed_data = crop.transform(input_data, sampling_rate_hz=1).transformed_data_

        expected_output = np.array([[4, 5, 6]])

        assert transformed_data.shape == expected_output.shape
        np.testing.assert_array_equal(transformed_data, expected_output)

    def test_default_reflect(self):
        crop = Crop(crop_len_s=1.0)

        input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        transformed_data = crop.transform(input_data, sampling_rate_hz=1).transformed_data_

        expected_output = np.array([[4, 5, 6]])

        assert transformed_data.shape == expected_output.shape
        np.testing.assert_array_equal(transformed_data, expected_output)

    def test_index_cropping(self):
        crop = Crop(crop_len_s=1.0)

        input_data = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        transformed_data = crop.transform(input_data, sampling_rate_hz=1).transformed_data_

        expected_output = input_data.iloc[1:-1]

        pd.testing.assert_frame_equal(expected_output, transformed_data)

    def test_invalid_tuple_length(self):
        crop = Crop(crop_len_s=(1.0, 2.0, 3.0))

        input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(ValueError):
            crop.transform(input_data, sampling_rate_hz=1)

    def test_no_sampling_rate(self):
        crop = Crop(crop_len_s=1.0)

        input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(ValueError):
            crop.transform(input_data, sampling_rate_hz=None)

    def test_padding_longer_than_data(self):
        crop = Crop(crop_len_s=(2, 2))

        input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(ValueError):
            crop.transform(input_data, sampling_rate_hz=1)
