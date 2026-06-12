import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from scipy.spatial.transform import Rotation
from tpcp.testing import TestAlgorithmMixin
from typing_extensions import Self

from mobgap.consts import BF_SENSOR_COLS, SF_SENSOR_COLS
from mobgap.data import LabExampleDataset
from mobgap.gait_sequences import GsdIluz, GsdIluzAdaptiveGravity
from mobgap.orientation_estimation.base import BaseOrientationEstimation
from mobgap.utils.conversions import to_body_frame


class TestMetaGsdIluz(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = GsdIluz

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().detect(
            pd.DataFrame(np.zeros((1000, 6)), columns=BF_SENSOR_COLS), sampling_rate_hz=40.0
        )


class TestMetaGsdIluzAdaptiveGravity(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = GsdIluzAdaptiveGravity

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().detect(
            pd.DataFrame(np.zeros((1000, 6)), columns=SF_SENSOR_COLS), sampling_rate_hz=40.0
        )


class TestGsdIluz:
    """Tests for GsdIluz.

    Note, we don't test the influence of any single parameter here.
    We don't even really know, how they all influence the results.
    We just test the happy path and some potential edegecases.
    If people run into bugs when changing parameters, we can add more tests.
    """

    parameters: dict[str, any]

    @pytest.fixture(autouse=True, params=[GsdIluz.PredefinedParameters.original, GsdIluz.PredefinedParameters.updated])
    def set_parameters(self, request):
        self.parameters = request.param

    def test_no_gsds(self):
        data = pd.DataFrame(np.zeros((1000, 6)), columns=BF_SENSOR_COLS)

        output = GsdIluz().detect(data, sampling_rate_hz=40.0).gs_list_

        assert_frame_equal(output, pd.DataFrame(columns=["start", "end", "gs_id"]).astype("int64").set_index("gs_id"))

    def test_single_gsd(self):
        data = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2").data_ss

        output = GsdIluz().detect(to_body_frame(data), sampling_rate_hz=100.0).gs_list_

        assert len(output) == 1
        assert set(output.columns) == {"start", "end"}


class TestGsdIluzAdaptiveGravity:
    def test_sensor_frame_input_rejects_body_frame_axis(self):
        data = pd.DataFrame(np.zeros((1000, 6)), columns=SF_SENSOR_COLS)

        with pytest.raises(ValueError, match="Sensor-frame data requires"):
            GsdIluzAdaptiveGravity(expected_pa_axis="pa").detect(data, sampling_rate_hz=40.0)

    def test_no_gsds(self):
        data = pd.DataFrame(np.zeros((1000, 6)), columns=SF_SENSOR_COLS)

        output = GsdIluzAdaptiveGravity().detect(data, sampling_rate_hz=40.0).gs_list_

        assert_frame_equal(output, pd.DataFrame(columns=["start", "end", "gs_id"]).astype("int64").set_index("gs_id"))

    def test_single_gsd_matches_body_frame_iluz(self):
        data = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2").data_ss

        expected = GsdIluz().detect(to_body_frame(data), sampling_rate_hz=100.0).gs_list_
        output = GsdIluzAdaptiveGravity().detect(data, sampling_rate_hz=100.0).gs_list_

        assert_frame_equal(output, expected)

    def test_body_frame_input_matches_body_frame_iluz(self):
        data = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2").data_ss
        body_data = to_body_frame(data)

        expected = GsdIluz().detect(body_data, sampling_rate_hz=100.0).gs_list_
        output = GsdIluzAdaptiveGravity(expected_pa_axis="pa").detect(body_data, sampling_rate_hz=100.0).gs_list_

        assert_frame_equal(output, expected)

    def test_body_frame_input_requires_pa_axis(self):
        data = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2").data_ss
        body_data = to_body_frame(data)

        with pytest.raises(ValueError, match="Body-frame data requires"):
            GsdIluzAdaptiveGravity().detect(body_data, sampling_rate_hz=100.0)

    def test_detects_with_gravity_on_different_sensor_axis(self):
        data = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2").data_ss
        rotated_data = _rotate_sensor_axes_around_z(data)

        expected = GsdIluzAdaptiveGravity().detect(data, sampling_rate_hz=100.0).gs_list_
        output = GsdIluzAdaptiveGravity().detect(rotated_data, sampling_rate_hz=100.0).gs_list_

        assert_frame_equal(output, expected)

    def test_expected_pa_axis_uses_raw_sensor_axis(self):
        data = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2").data_ss
        rotated_data = _rotate_sensor_axes_around_x(data)

        output = GsdIluzAdaptiveGravity(expected_pa_axis="y").detect(rotated_data, sampling_rate_hz=100.0)

        assert len(output.gs_list_) == 1
        assert_series_equal(output.iluz_data_["acc_pa"], rotated_data["acc_y"], check_names=False)

    def test_sensor_frame_input_rejects_body_frame_axis_on_real_data(self):
        data = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2").data_ss

        with pytest.raises(ValueError, match="Sensor-frame data requires"):
            GsdIluzAdaptiveGravity(expected_pa_axis="pa").detect(data, sampling_rate_hz=100.0)

    def test_body_frame_input_uses_raw_pa_axis(self):
        data = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2").data_ss
        body_data = to_body_frame(data)

        output = GsdIluzAdaptiveGravity(expected_pa_axis="pa").detect(body_data, sampling_rate_hz=100.0)

        assert_series_equal(output.iluz_data_["acc_pa"], body_data["acc_pa"], check_names=False)

    def test_body_frame_input_can_use_other_raw_body_axis(self):
        data = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2").data_ss
        body_data = to_body_frame(data)

        output = GsdIluzAdaptiveGravity(expected_pa_axis="ml").detect(body_data, sampling_rate_hz=100.0)

        assert_series_equal(output.iluz_data_["acc_pa"], body_data["acc_ml"], check_names=False)

    def test_orientation_estimation_parameter_is_used(self):
        data = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2").data_ss

        output = GsdIluzAdaptiveGravity(orientation_estimation=_IdentityOrientationEstimation()).detect(
            data, sampling_rate_hz=100.0
        )

        assert_series_equal(output.iluz_data_["acc_is"], data["acc_z"], check_names=False)
        assert not hasattr(output, "orientation_object_")


class TestGsdIluzRegression:
    @pytest.mark.parametrize("datapoint", LabExampleDataset(reference_system="INDIP", reference_para_level="wb"))
    @pytest.mark.parametrize("use_original", [True, False])
    def test_example_lab_data(self, datapoint, snapshot, use_original):
        data = datapoint.data_ss
        sampling_rate_hz = datapoint.sampling_rate_hz
        parameters = GsdIluz.PredefinedParameters.original if use_original else GsdIluz.PredefinedParameters.updated

        gs_list = GsdIluz(**parameters).detect(to_body_frame(data), sampling_rate_hz=sampling_rate_hz).gs_list_
        snapshot.assert_match(gs_list, str(tuple(datapoint.group_label)))


def _rotate_sensor_axes_around_z(data: pd.DataFrame) -> pd.DataFrame:
    rotated_data = data.copy()
    for sensor in ("acc", "gyr"):
        old_x = rotated_data[f"{sensor}_x"].copy()
        old_y = rotated_data[f"{sensor}_y"].copy()
        rotated_data[f"{sensor}_x"] = -old_y
        rotated_data[f"{sensor}_y"] = old_x
    return rotated_data


def _rotate_sensor_axes_around_x(data: pd.DataFrame) -> pd.DataFrame:
    rotated_data = data.copy()
    for sensor in ("acc", "gyr"):
        old_y = rotated_data[f"{sensor}_y"].copy()
        old_z = rotated_data[f"{sensor}_z"].copy()
        rotated_data[f"{sensor}_y"] = old_z
        rotated_data[f"{sensor}_z"] = -old_y
    return rotated_data


class _IdentityOrientationEstimation(BaseOrientationEstimation):
    def estimate(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_) -> Self:
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.orientation_object_ = Rotation.from_quat(np.repeat([[0.0, 0.0, 0.0, 1.0]], len(data) + 1, axis=0))
        return self
