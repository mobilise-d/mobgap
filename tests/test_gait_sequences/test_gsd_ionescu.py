import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.data import LabExampleDataset
from mobgap.gait_sequences import GsdAdaptiveIonescu
from mobgap.gait_sequences._gsd_ionescu import GsdIonescu, find_intersections


class TestMetaGsdParaschivIonescu(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = GsdAdaptiveIonescu

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().detect(
            pd.DataFrame(np.zeros((1000, 3)), columns=["acc_x", "acc_y", "acc_z"]), sampling_rate_hz=40.0
        )


class TestIntersect:
    def test_non_overlapping_intervals(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        expected = []
        result = find_intersections(a, b)
        assert np.array_equal(result, expected), "Should return an empty array for non-overlapping intervals"

    def test_overlapping_intervals(self):
        a = np.array([[1, 5], [6, 10]])
        b = np.array([[4, 7], [8, 12]])
        expected = np.array([[4, 5], [6, 7], [8, 10]])
        result = find_intersections(a, b)
        assert np.array_equal(
            result[: len(expected)], expected
        ), "Should return correct intersections for overlapping intervals"


class TestGsdIonescu:
    """Tests for GsdAdaptiveIonescu.

    Note, we don't test the influence of any single parameter here.
    We don't even really know, how they all influence the results.
    We just test the happy path and some potential edege cases.
    If people run into bugs when changing parameters, we can add more tests.
    """

    @pytest.fixture(autouse=True, params=[GsdIonescu, GsdAdaptiveIonescu])
    def algorithm_instance(self, request):
        self.algorithm = request.param

    def test_no_gsds(self):
        data = pd.DataFrame(np.zeros((1000, 3)), columns=["acc_x", "acc_y", "acc_z"])

        output = self.algorithm().detect(data, sampling_rate_hz=40.0).gs_list_

        assert_frame_equal(output, pd.DataFrame(columns=["start", "end", "gs_id"]).astype("int64").set_index("gs_id"))

    def test_single_gsd(self):
        data = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2").data_ss

        output = self.algorithm().detect(data, sampling_rate_hz=100.0).gs_list_

        assert len(output) == 1
        assert set(output.columns) == {"start", "end"}

    def test_random_signal(self):
        data = pd.DataFrame(np.random.rand(1000, 3), columns=["acc_x", "acc_y", "acc_z"])

        output = self.algorithm().detect(data, sampling_rate_hz=100.0).gs_list_

        assert set(output.columns) == {"start", "end"}


class TestGsdAdaptiveIonescuRegression:
    @pytest.mark.parametrize("datapoint", LabExampleDataset(reference_system="INDIP", reference_para_level="wb"))
    def test_example_lab_data(self, datapoint, snapshot):
        data = datapoint.data_ss
        sampling_rate_hz = datapoint.sampling_rate_hz

        gs_list = GsdAdaptiveIonescu().detect(data, sampling_rate_hz=sampling_rate_hz).gs_list_
        snapshot.assert_match(gs_list, str(tuple(datapoint.group_label)))


class TestGsdIonescuRegression:
    @pytest.mark.parametrize("datapoint", LabExampleDataset(reference_system="INDIP", reference_para_level="wb"))
    def test_example_lab_data(self, datapoint, snapshot):
        data = datapoint.data_ss
        sampling_rate_hz = datapoint.sampling_rate_hz

        gs_list = GsdIonescu().detect(data, sampling_rate_hz=sampling_rate_hz).gs_list_
        snapshot.assert_match(gs_list, str(tuple(datapoint.group_label)))
