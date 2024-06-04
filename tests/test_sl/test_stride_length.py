import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.sl import SlZijlstra
from mobgap.data import LabExampleDataset
from mobgap.pipeline import GsIterator

class TestMetaSlZijlstra(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = SlZijlstra

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().calculate(
            data = pd.DataFrame(np.zeros((100, 3)), columns=["acc_x", "acc_y", "acc_z"]),
            initial_contacts=pd.DataFrame({"ic": np.arange(0, 100, 5)}),
            sensor_height_m= 0.95,
            sampling_rate_hz=100.0,
        )

class TestSlZijlstra:
    """Tests for SlZijlstra.

    We just test the happy path and some potential edgecases.
    If people run into bugs when changing parameters, we can add more tests.
    """
    def test_not_enough_ics(self):
        data = pd.DataFrame(np.zeros((100, 3)), columns=["acc_x", "acc_y", "acc_z"])
        initial_contacts = pd.DataFrame({"ic": np.arange(0, 100, 5)})
        # We only keep the first IC -> Not possible to calculate step length
        initial_contacts = initial_contacts.iloc[:1]

        with pytest.warns(UserWarning) as w:
            sl_zijlstra = SlZijlstra().calculate(data = data, initial_contacts = initial_contacts, sensor_height_m = 0.95, sampling_rate_hz=100)

        assert len(w) == 1
        assert "Can not calculate step length with only one or zero initial contacts" in w.list[0].message.args[0]
        assert len(sl_zijlstra.stride_length_per_sec_list_) == np.floor(data.shape[0] / 100)
        assert sl_zijlstra.stride_length_per_sec_list_["stride_length_m"].isna().all()

    def test_raise_non_sorted_ics(self):
        data = pd.DataFrame(np.zeros((100, 3)), columns=["acc_x", "acc_y", "acc_z"])
        initial_contacts = pd.DataFrame({"ic": np.arange(0, 100, 5)})
        # We shuffle the ICs
        initial_contacts = initial_contacts.sample(frac=1, random_state=2)

        with pytest.raises(ValueError) as e:
            SlZijlstra().calculate(data = data, initial_contacts = initial_contacts, sensor_height_m = 0.95, sampling_rate_hz=40.0)

        assert "Initial contacts must be sorted" in str(e.value)