import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.data import LabExampleDataset
from mobgap.icd import IcdIonescu
from mobgap.pipeline import GsIterator


class TestMetaIcdIonescu(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = IcdIonescu

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().detect(
            pd.DataFrame(np.zeros((1000, 3)), columns=["acc_x", "acc_y", "acc_z"]), sampling_rate_hz=40.0
        )


class TestIcdIonescu:
    """Tests for IcdIonescu.

    We just test the happy path and some potential edgecases.
    If people run into bugs when changing parameters, we can add more tests.
    """

    def test_no_icds(self):
        data = pd.DataFrame(np.zeros((1000, 3)), columns=["acc_x", "acc_y", "acc_z"])  # not a gait sequence

        output = IcdIonescu().detect(data, sampling_rate_hz=100.0).ic_list_

        assert_frame_equal(output, pd.DataFrame({"ic": []}).rename_axis(index="step_id").astype("int64"))

    def test_single_icd(self):
        # s and e delimit a gait sequence with just one IC
        s = 700
        e = 800
        data = (
            LabExampleDataset()
            .get_subset(cohort="MS", participant_id="001", test="Test5", trial="Trial1")
            .data["LowerBack"][s : e + 1]
        )

        output = IcdIonescu().detect(data, sampling_rate_hz=100.0).ic_list_

        assert len(output) == 1
        assert output.columns == ["ic"]


class TestIcdIonescuRegression:
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
            result.ic_list = IcdIonescu().detect(data, sampling_rate_hz=sampling_rate_hz).ic_list_

        detected_ics = iterator.results_.ic_list
        snapshot.assert_match(detected_ics, str(datapoint.group_label))
