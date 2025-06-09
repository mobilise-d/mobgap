import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS
from mobgap.data import LabExampleDataset
from mobgap.initial_contacts import IcdIonescu
from mobgap.pipeline import GsIterator
from mobgap.utils.conversions import to_body_frame


class TestMetaIcdIonescu(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = IcdIonescu

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().detect(
            pd.DataFrame(np.zeros((1000, 6)), columns=BF_SENSOR_COLS), sampling_rate_hz=40.0
        )


class TestIcdIonescu:
    """Tests for IcdIonescu.

    We just test the happy path and some potential edgecases.
    If people run into bugs when changing parameters, we can add more tests.
    """

    def test_no_icds(self):
        data = pd.DataFrame(np.zeros((1000, 6)), columns=BF_SENSOR_COLS)  # not a gait sequence

        output = IcdIonescu().detect(data, sampling_rate_hz=100.0).ic_list_

        assert_frame_equal(output, pd.DataFrame({"ic": []}).rename_axis(index="step_id").astype("int64"))

    def test_single_icd(self):
        # s and e delimit a gait sequence with just one IC
        s = 700
        e = 800
        data = (
            LabExampleDataset()
            .get_subset(cohort="MS", participant_id="001", test="Test5", trial="Trial1")
            .data_ss[s : e + 1]
        )

        output = IcdIonescu().detect(to_body_frame(data), sampling_rate_hz=100.0).ic_list_

        assert len(output) == 1
        assert output.columns == ["ic"]


class TestIcdIonescuRegression:
    @pytest.mark.parametrize("datapoint", LabExampleDataset(reference_system="INDIP", reference_para_level="wb"))
    def test_example_lab_data(self, datapoint, snapshot):
        data = to_body_frame(datapoint.data_ss)
        ref_walk_bouts = datapoint.reference_parameters_.wb_list
        if len(ref_walk_bouts) == 0:
            pytest.skip("No reference parameters available.")
        sampling_rate_hz = datapoint.sampling_rate_hz

        iterator = GsIterator()

        for (gs, data), result in iterator.iterate(data, ref_walk_bouts):
            result.ic_list = IcdIonescu().detect(data, sampling_rate_hz=sampling_rate_hz).ic_list_

        detected_ics = iterator.results_.ic_list
        snapshot.assert_match(detected_ics, str(tuple(datapoint.group_label)))
