import numpy as np
import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from gaitlink.data import LabExampleDataset
from gaitlink.icd._hklee_algo_improved import IcdHKLeeImproved
from gaitlink.pipeline import GsIterator


class TestMetaHKLeeImproved(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = IcdHKLeeImproved

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().detect(
            pd.DataFrame(np.zeros((1000, 3)), columns=["acc_x", "acc_y", "acc_z"]), sampling_rate_hz=120.0
        )


class TestHKLeeImproved:
    def test_invalid_axis_parameter(self):
        with pytest.raises(ValueError):
            IcdHKLeeImproved(axis="invalid").detect(pd.DataFrame(), sampling_rate_hz=100)

    def test_no_ics_detected(self):
        data = pd.DataFrame(np.zeros((1000, 3)), columns=["acc_x", "acc_y", "acc_z"])
        output = IcdHKLeeImproved(axis="x")
        output.detect(data, sampling_rate_hz=120.0)
        output_ic = output.ic_list_["ic"]
        empty_output = {}
        assert output_ic.to_dict() == empty_output


class TestHKLeeImprovedRegression:
    @pytest.mark.parametrize("datapoint", LabExampleDataset(reference_system="INDIP", reference_para_level="wb"))
    def test_example_lab_data(self, datapoint, snapshot):
        data = datapoint.data["LowerBack"]
        try:
            ref_walk_bouts = datapoint.reference_parameters_.wb_list
        except:
            pytest.skip("No reference parameters available.")
        sampling_rate_hz = datapoint.sampling_rate_hz

        iterator = GsIterator()

        for (gs, data), result in iterator.iterate(data, ref_walk_bouts):
            result.ic_list = IcdHKLeeImproved().detect(data, sampling_rate_hz=sampling_rate_hz).ic_list_

        detected_ics = iterator.ic_list_
        snapshot.assert_match(detected_ics, str(datapoint.group_label))
