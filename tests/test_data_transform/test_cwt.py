import pytest
import numpy as np
from scipy.signal import morlet
from tpcp.testing import TestAlgorithmMixin
from gaitlink.data_transform._cwt_filter import CwtFilter


class TestMetaCwtFilter(TestAlgorithmMixin):
    ALGORITHM_CLASS = CwtFilter
    __test__ = True
    ONLY_DEFAULT_PARAMS = False

    @pytest.fixture()
    def after_action_instance(self):
        # Creating an instance of your CwtFilter class with some initial conditions
        cwt_filter_instance = self.ALGORITHM_CLASS(wavelet=morlet, width=np.arange(1, 31))

        # Use some random input data for the meta-test (actual values might not be important)
        input_data = np.array([1.0, 2.0, 3.0, 4.0])

        # Perform the Continuous Wavelet Transform using your CwtFilter class
        result = cwt_filter_instance.transform(input_data, widths=[1, 2, 3])

        # Return the CwtFilter class instance with initial conditions
        return cwt_filter_instance
