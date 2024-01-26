import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from gaitlink.data import LabExampleDataset
from gaitlink.icd import IcdShinImproved
from gaitlink.pipeline import GsIterator


class TestMetaShinImproved(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = IcdShinImproved

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().detect(
            pd.DataFrame(np.zeros((1000, 3)), columns=["acc_x", "acc_y", "acc_z"]), sampling_rate_hz=40.0
        )


class TestShinImproved:
    def test_invalid_axis_parameter(self):
        with pytest.raises(ValueError):
            IcdShinImproved(axis="invalid").detect(pd.DataFrame(), sampling_rate_hz=100)

    def test_no_ics_detected(self):
        data = pd.DataFrame(np.zeros((1000, 3)), columns=["acc_x", "acc_y", "acc_z"])
        output = IcdShinImproved(axis="x")
        output.detect(data, sampling_rate_hz=40.0)
        output_ic = output.ic_list_["ic"]
        empty_output = {}
        assert output_ic.to_dict() == empty_output


class TestShinImprovedRegression:
    @pytest.mark.parametrize("datapoint", LabExampleDataset(reference_system="INDIP", reference_para_level="wb"))
    def test_example_lab_data(self, datapoint, snapshot):
        data = datapoint.data["LowerBack"]
        try:
            ref_walk_bouts = datapoint.reference_parameters_.walking_bouts
        except:
            pytest.skip("No reference parameters available.")
        sampling_rate_hz = datapoint.sampling_rate_hz

        iterator = GsIterator()

        for (gs, data), result in iterator.iterate(data, ref_walk_bouts):
            result.initial_contacts = IcdShinImproved().detect(data, sampling_rate_hz=sampling_rate_hz).ic_list_

        detected_ics = iterator.initial_contacts_
        snapshot.assert_match(detected_ics, str(datapoint.group_label))
