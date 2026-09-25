import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_ACC_COLS, BF_SENSOR_COLS, SF_ACC_COLS, SF_SENSOR_COLS
from mobgap.data import LabExampleDataset
from mobgap.initial_contacts import IcdShinImproved
from mobgap.pipeline import GsIterator
from mobgap.utils.conversions import to_body_frame


class TestMetaShinImproved(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = IcdShinImproved

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().detect(
            pd.DataFrame(np.zeros((1000, 6)), columns=BF_SENSOR_COLS), sampling_rate_hz=40.0
        )


class TestShinImproved:
    @pytest.mark.parametrize(
        "axis,all_columns,required_columns",
        [
            ("is", BF_SENSOR_COLS, ["acc_is"]),
            ("norm", BF_SENSOR_COLS, BF_ACC_COLS),
            ("norm", SF_SENSOR_COLS, SF_ACC_COLS),
        ],
    )
    def test_required_acceleration_only_matches_full_input(self, axis, all_columns, required_columns):
        data = pd.DataFrame(np.zeros((1000, 6)), columns=all_columns)

        expected = IcdShinImproved(axis=axis).detect(data, sampling_rate_hz=40.0).ic_list_
        actual = IcdShinImproved(axis=axis).detect(data[required_columns], sampling_rate_hz=40.0).ic_list_

        assert_frame_equal(actual, expected)

    def test_invalid_axis_parameter(self):
        with pytest.raises(ValueError):
            IcdShinImproved(axis="invalid").detect(pd.DataFrame(), sampling_rate_hz=100)

    def test_no_ics_detected(self):
        data = pd.DataFrame(np.zeros((1000, 6)), columns=BF_SENSOR_COLS)
        output = IcdShinImproved(axis="is")
        output.detect(data, sampling_rate_hz=40.0)
        output_ic = output.ic_list_["ic"]
        empty_output = {}
        assert output_ic.to_dict() == empty_output


class TestShinImprovedRegression:
    @pytest.mark.parametrize("datapoint", LabExampleDataset(reference_system="INDIP", reference_para_level="wb"))
    def test_example_lab_data(self, datapoint, snapshot):
        data = to_body_frame(datapoint.data_ss)
        ref_walk_bouts = datapoint.reference_parameters_.wb_list
        if len(ref_walk_bouts) == 0:
            pytest.skip("No reference parameters available.")
        sampling_rate_hz = datapoint.sampling_rate_hz

        iterator = GsIterator()

        for (gs, data), result in iterator.iterate(data, ref_walk_bouts):
            result.ic_list = IcdShinImproved().detect(data, sampling_rate_hz=sampling_rate_hz).ic_list_

        detected_ics = iterator.results_.ic_list
        snapshot.assert_match(detected_ics, str(tuple(datapoint.group_label)))
