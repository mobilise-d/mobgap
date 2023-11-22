import pandas as pd
import pytest
from scipy.signal import savgol_filter
from tpcp.testing import TestAlgorithmMixin
from gaitlink.data_transform._savgol_filter import SavgolFilter

class TestMetaSavgolFilter(TestAlgorithmMixin):
    ALGORITHM_CLASS = SavgolFilter
    __test__ = True

    @pytest.fixture()
    def after_action_instance(self):
        # Creating an instance of your SavgolFilter class with some initial conditions
        savgol_filter_instance = self.ALGORITHM_CLASS(window_length=5, polyorder=2)

        # Use some random input data for the meta-test (actual values might not be important)
        input_data = pd.DataFrame({
            "sensor1": [1.0, 2.0, 3.0, 4.0],
            "sensor2": [0.5, 1.0, 1.5, 2.0],
        })

        # Perform the Savitzky-Golay filtering using your SavgolFilter class
        result = savgol_filter_instance.transform(input_data)

        # Return the SavgolFilter class instance with initial conditions
        return savgol_filter_instance
