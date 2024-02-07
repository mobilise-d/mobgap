import numpy as np
import pytest
from tpcp.testing import TestAlgorithmMixin

from gaitlink.data_transform import Crop, Pad


class TestMetaPad(TestAlgorithmMixin):
    ALGORITHM_CLASS = Pad
    ONLY_DEFAULT_PARAMS = False
    __test__ = True

    @pytest.fixture()
    def after_action_instance(self):
        # Creating an instance of your Pad class with some initial conditions
        pad = self.ALGORITHM_CLASS(pad_len_s=1.0)

        # Use some random input data for the meta-test (actual values might not be important)
        input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Perform the padding action using your Pad class
        pad.transform(input_data, sampling_rate_hz=100)

        # Return the Pad class instance with initial conditions
        return pad


class TestMetaCrop(TestAlgorithmMixin):
    ALGORITHM_CLASS = Crop
    ONLY_DEFAULT_PARAMS = False
    __test__ = True

    @pytest.fixture()
    def after_action_instance(self):
        # Creating an instance of your Crop class with some initial conditions
        crop = self.ALGORITHM_CLASS(crop_len_s=1.0)

        # Use some random input data for the meta-test (actual values might not be important)
        input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Perform the cropping action using your Crop class
        crop.transform(input_data, sampling_rate_hz=1)

        # Return the Crop class instance with initial conditions
        return crop
