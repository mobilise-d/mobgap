import warnings

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import SF_SENSOR_COLS
from mobgap.data import LabExampleDataset
from mobgap.data_transform.base import BaseFilter
from mobgap.pipeline import GsIterator
from mobgap.re_orientation import ReorientationMethodDM
from mobgap.re_orientation.evaluation import reorientation_score
from mobgap.re_orientation.pipeline import (
    REORIENTATION_LABELS,
    ReorientationEmulationPipeline,
)
from mobgap.utils.conversions import to_body_frame
from mobgap.utils.evaluation import Evaluation


class _ZeroPhaseFilter(BaseFilter):
    def filter(self, data, *, sampling_rate_hz=None, **kwargs):
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.transformed_data_ = pd.DataFrame(
            {"acc_x": data["acc_z"].to_numpy(), "acc_z": data["acc_z"].to_numpy()}, index=data.index
        )
        return self


class TestMetaReorientationMethodDM(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = ReorientationMethodDM

    @pytest.fixture
    def after_action_instance(self):
        algo = self.ALGORITHM_CLASS(gravity_detection_error_type="ignore")
        algo.detect_correct(pd.DataFrame(np.zeros((1000, 6)), columns=SF_SENSOR_COLS), sampling_rate_hz=100.0)
        return algo


class TestReorientationMethodDM:
    """Tests for ReorientationMethodDM."""

    def test_sensor_frame_input_returns_body_frame_output(self):
        data = pd.DataFrame(
            {
                "acc_x": np.ones(1000) * 9.8,
                "acc_y": np.arange(1000, dtype=float),
                "acc_z": np.arange(1000, dtype=float) + 1000,
                "gyr_x": np.arange(1000, dtype=float) + 2000,
                "gyr_y": np.arange(1000, dtype=float) + 3000,
                "gyr_z": np.arange(1000, dtype=float) + 4000,
            }
        )

        result = ReorientationMethodDM(correction_mode="trust_gravity").detect_correct(data, sampling_rate_hz=100.0)

        expected = to_body_frame(data)
        assert_frame_equal(result.corrected_data_, expected)

    def test_default_correction_mode_is_trust_gravity(self):
        result = ReorientationMethodDM().detect_correct(
            pd.DataFrame(
                {
                    "acc_x": np.ones(10) * 9.8,
                    "acc_y": np.zeros(10),
                    "acc_z": np.zeros(10),
                    "gyr_x": np.zeros(10),
                    "gyr_y": np.zeros(10),
                    "gyr_z": np.zeros(10),
                }
            ),
            sampling_rate_hz=100.0,
        )

        assert result.correction_mode == "trust_gravity"
        assert result.result_.phase is None
        assert result.result_.orientation_resolved is True

    def test_uses_sampling_rate_for_phase_detection(self):
        sampling_rate_hz = 50.0
        gait_frequency_hz = 1.8
        time_s = np.arange(int(20 * sampling_rate_hz)) / sampling_rate_hz
        base = pd.DataFrame(
            {
                "acc_x": 9.81 + np.sin(2 * np.pi * gait_frequency_hz * time_s),
                "acc_y": 0.2 * np.sin(2 * np.pi * gait_frequency_hz * time_s + 0.4),
                "acc_z": np.cos(2 * np.pi * gait_frequency_hz * time_s),
                "gyr_x": 0.01 * np.sin(2 * np.pi * gait_frequency_hz * time_s),
                "gyr_y": 0.02 * np.cos(2 * np.pi * gait_frequency_hz * time_s),
                "gyr_z": 0.03 * np.sin(2 * np.pi * gait_frequency_hz * time_s + 0.1),
            }
        )
        data = base.copy()
        data[["acc_x", "gyr_x", "acc_y", "gyr_y"]] *= -1

        result = ReorientationMethodDM(correction_mode="full").detect_correct(data, sampling_rate_hz=sampling_rate_hz)

        assert_frame_equal(result.corrected_data_, to_body_frame(base))

    def test_correctly_oriented_data_family_is_up(self):
        """Test that is_up data is left unchanged in trust-gravity mode."""
        data = pd.DataFrame(
            {
                "acc_x": np.ones(1000) * 9.8,
                "acc_y": np.random.randn(1000) * 0.5,
                "acc_z": np.random.randn(1000) * 0.5,
                "gyr_x": np.random.randn(1000) * 0.1,
                "gyr_y": np.random.randn(1000) * 0.1,
                "gyr_z": np.random.randn(1000) * 0.1,
            }
        )

        result = ReorientationMethodDM(correction_mode="trust_gravity").detect_correct(data, sampling_rate_hz=100.0)

        assert result.result_.family == "is_up"
        assert result.result_.where_grav == "x"
        assert result.result_.where_grav_points == "up"

    def test_family_is_down_misorientation(self):
        """Test is_down (IS pointing down) is corrected."""
        data = pd.DataFrame(
            {
                "acc_x": np.ones(1000) * -9.8,
                "acc_y": np.random.randn(1000) * 0.5,
                "acc_z": np.random.randn(1000) * 0.5,
                "gyr_x": np.random.randn(1000) * 0.1,
                "gyr_y": np.random.randn(1000) * 0.1,
                "gyr_z": np.random.randn(1000) * 0.1,
            }
        )

        result = ReorientationMethodDM(correction_mode="full").detect_correct(data, sampling_rate_hz=100.0)

        assert result.result_.family == "is_down"
        assert result.result_.where_grav == "x"
        assert result.result_.where_grav_points == "down"
        assert result.result_.correction_applied is True
        assert "rotated 180 deg around sensor z-axis" in result.result_.correction_action

    def test_family_ml_up_misorientation(self):
        """Test ml_up (gravity in ML pointing up) is corrected."""
        data = pd.DataFrame(
            {
                "acc_x": np.random.randn(1000) * 0.5,
                "acc_y": np.ones(1000) * 9.8,
                "acc_z": np.random.randn(1000) * 0.5,
                "gyr_x": np.random.randn(1000) * 0.1,
                "gyr_y": np.random.randn(1000) * 0.1,
                "gyr_z": np.random.randn(1000) * 0.1,
            }
        )

        result = ReorientationMethodDM(correction_mode="full").detect_correct(data, sampling_rate_hz=100.0)

        assert result.result_.family == "ml_up"
        assert result.result_.where_grav == "y"
        assert result.result_.where_grav_points == "up"
        assert result.result_.correction_applied is True
        assert "rotated -90 deg around sensor z-axis" in result.result_.correction_action

    def test_family_ml_down_misorientation(self):
        """Test ml_down (gravity in ML pointing down) is corrected."""
        data = pd.DataFrame(
            {
                "acc_x": np.random.randn(1000) * 0.5,
                "acc_y": np.ones(1000) * -9.8,
                "acc_z": np.random.randn(1000) * 0.5,
                "gyr_x": np.random.randn(1000) * 0.1,
                "gyr_y": np.random.randn(1000) * 0.1,
                "gyr_z": np.random.randn(1000) * 0.1,
            }
        )

        result = ReorientationMethodDM(correction_mode="full").detect_correct(data, sampling_rate_hz=100.0)

        assert result.result_.family == "ml_down"
        assert result.result_.where_grav == "y"
        assert result.result_.where_grav_points == "down"
        assert result.result_.correction_applied is True
        assert "rotated 90 deg around sensor z-axis" in result.result_.correction_action

    def test_no_gravity_detected_warns_and_returns_uncorrected_data(self):
        """Test that data without clear gravity signal returns None family."""
        data = pd.DataFrame(
            {
                "acc_x": np.random.randn(1000) * 2.0,
                "acc_y": np.random.randn(1000) * 2.0,
                "acc_z": np.random.randn(1000) * 2.0,
                "gyr_x": np.random.randn(1000) * 0.1,
                "gyr_y": np.random.randn(1000) * 0.1,
                "gyr_z": np.random.randn(1000) * 0.1,
            }
        )

        with pytest.warns(UserWarning, match="No sensor axis with a clear gravity signal could be identified"):
            result = ReorientationMethodDM(correction_mode="full").detect_correct(data, sampling_rate_hz=100.0)

        assert result.result_.family is None
        assert result.result_.where_grav is None
        assert result.result_.orientation_resolved is False
        assert result.result_.unresolved_reason == "gravity"
        assert result.result_.correction_applied is False
        assert result.result_.correction_action == "none"
        assert_frame_equal(result.result_.data_corrected, to_body_frame(data))

    def test_no_gravity_detected_can_raise(self):
        data = pd.DataFrame(np.zeros((1000, 6)), columns=SF_SENSOR_COLS)

        with pytest.raises(ValueError, match="No sensor axis with a clear gravity signal could be identified"):
            ReorientationMethodDM(gravity_detection_error_type="raise").detect_correct(data, sampling_rate_hz=100.0)

    def test_no_gravity_detected_can_be_ignored(self):
        data = pd.DataFrame(np.zeros((1000, 6)), columns=SF_SENSOR_COLS)

        with warnings.catch_warnings(record=True) as warning_info:
            result = ReorientationMethodDM(gravity_detection_error_type="ignore").detect_correct(
                data, sampling_rate_hz=100.0
            )

        assert len(warning_info) == 0
        assert result.result_.unresolved_reason == "gravity"

    def test_full_vs_trust_gravity_correction_mode_family_is_up(self):
        """Test that full and trust-gravity modes differ for is_up family."""
        single_test = LabExampleDataset(reference_system="INDIP", reference_para_level="wb").get_subset(
            cohort="HA", participant_id="001", test="Test11", trial="Trial1"
        )

        reference_wbs = single_test.reference_parameters_.wb_list
        start = reference_wbs.iloc[2]["start"]
        end = reference_wbs.iloc[2]["end"]

        imu_data = single_test.data.get("LowerBack")
        imu_data = imu_data.reset_index(drop=True)
        wb_data = imu_data.loc[start:end]

        result_full = ReorientationMethodDM(correction_mode="full").detect_correct(
            wb_data, sampling_rate_hz=single_test.sampling_rate_hz
        )
        result_trust_gravity = ReorientationMethodDM(correction_mode="trust_gravity").detect_correct(
            wb_data, sampling_rate_hz=single_test.sampling_rate_hz
        )

        # Both should detect same family
        assert result_full.result_.family == result_trust_gravity.result_.family

        # If is_up, trust-gravity mode skips Stage 3
        if result_full.result_.family == "is_up":
            assert len(result_trust_gravity.result_.correction_action) <= len(result_full.result_.correction_action)

    def test_invalid_correction_mode_parameter(self):
        """Test that invalid correction mode parameter raises ValueError."""
        with pytest.raises(ValueError, match="correction_mode must be 'full' or 'trust_gravity'"):
            ReorientationMethodDM(correction_mode="invalid").detect_correct(
                pd.DataFrame(np.zeros((1000, 6)), columns=SF_SENSOR_COLS), sampling_rate_hz=100.0
            )

    def test_invalid_error_type_parameter(self):
        with pytest.raises(ValueError, match="gravity_detection_error_type must be one of"):
            ReorientationMethodDM(gravity_detection_error_type="invalid").detect_correct(
                pd.DataFrame(np.zeros((1000, 6)), columns=SF_SENSOR_COLS), sampling_rate_hz=100.0
            )

    def test_unresolved_phase_is_reported_as_none(self):
        data = pd.DataFrame(
            {
                "acc_x": np.ones(10) * 9.8,
                "acc_y": np.zeros(10),
                "acc_z": np.zeros(10),
                "gyr_x": np.zeros(10),
                "gyr_y": np.zeros(10),
                "gyr_z": np.zeros(10),
            }
        )

        with pytest.warns(UserWarning, match="The direction of the PA axis could not be resolved"):
            result = ReorientationMethodDM(correction_mode="full").detect_correct(data, sampling_rate_hz=100.0)

        assert result.result_.phase is None

    def test_zero_phase_is_reported_as_unresolved_pa_direction(self):
        sampling_rate_hz = 100.0
        time_s = np.arange(1000) / sampling_rate_hz
        data = pd.DataFrame(
            {
                "acc_x": np.ones(1000) * 9.8,
                "acc_y": np.zeros(1000),
                "acc_z": np.sin(2 * np.pi * 1.2 * time_s),
                "gyr_x": np.zeros(1000),
                "gyr_y": np.zeros(1000),
                "gyr_z": np.zeros(1000),
            }
        )

        with pytest.warns(UserWarning, match="The direction of the PA axis could not be resolved"):
            result = ReorientationMethodDM(
                correction_mode="full", gait_frequency_band_filter=_ZeroPhaseFilter()
            ).detect_correct(data, sampling_rate_hz=sampling_rate_hz)

        assert result.result_.phase is None
        assert result.result_.orientation_resolved is False
        assert result.result_.unresolved_reason == "pa_direction"
        assert_frame_equal(result.corrected_data_, to_body_frame(data))

    def test_unresolved_pa_direction_warns_and_assumes_pa_is_correct(self):
        data = pd.DataFrame(
            {
                "acc_x": [-9.8, -9.8],
                "acc_y": [1.0, 2.0],
                "acc_z": [3.0, 4.0],
                "gyr_x": [5.0, 6.0],
                "gyr_y": [7.0, 8.0],
                "gyr_z": [9.0, 10.0],
            }
        )

        with pytest.warns(UserWarning, match="The direction of the PA axis could not be resolved"):
            result = ReorientationMethodDM(correction_mode="full").detect_correct(data, sampling_rate_hz=100.0)

        expected = pd.DataFrame(
            {
                "acc_is": [9.8, 9.8],
                "acc_ml": [-1.0, -2.0],
                "acc_pa": [3.0, 4.0],
                "gyr_is": [-5.0, -6.0],
                "gyr_ml": [-7.0, -8.0],
                "gyr_pa": [9.0, 10.0],
            }
        )
        assert result.result_.phase is None
        assert result.result_.orientation_resolved is False
        assert result.result_.unresolved_reason == "pa_direction"
        assert_frame_equal(result.corrected_data_, expected)

    def test_unresolved_pa_direction_can_raise(self):
        data = pd.DataFrame(
            {
                "acc_x": np.ones(10) * 9.8,
                "acc_y": np.zeros(10),
                "acc_z": np.zeros(10),
                "gyr_x": np.zeros(10),
                "gyr_y": np.zeros(10),
                "gyr_z": np.zeros(10),
            }
        )

        with pytest.raises(ValueError, match="The direction of the PA axis could not be resolved"):
            ReorientationMethodDM(correction_mode="full", pa_direction_detection_error_type="raise").detect_correct(
                data, sampling_rate_hz=100.0
            )

    def test_unresolved_pa_direction_can_be_ignored(self):
        data = pd.DataFrame(
            {
                "acc_x": np.ones(10) * 9.8,
                "acc_y": np.zeros(10),
                "acc_z": np.zeros(10),
                "gyr_x": np.zeros(10),
                "gyr_y": np.zeros(10),
                "gyr_z": np.zeros(10),
            }
        )

        with warnings.catch_warnings(record=True) as warning_info:
            result = ReorientationMethodDM(
                correction_mode="full", pa_direction_detection_error_type="ignore"
            ).detect_correct(data, sampling_rate_hz=100.0)

        assert len(warning_info) == 0
        assert result.result_.unresolved_reason == "pa_direction"

    def test_trust_gravity_family_1_does_not_require_pa_direction(self):
        data = pd.DataFrame(
            {
                "acc_x": np.ones(10) * 9.8,
                "acc_y": np.zeros(10),
                "acc_z": np.zeros(10),
                "gyr_x": np.zeros(10),
                "gyr_y": np.zeros(10),
                "gyr_z": np.zeros(10),
            }
        )

        result = ReorientationMethodDM(
            correction_mode="trust_gravity", pa_direction_detection_error_type="raise"
        ).detect_correct(data, sampling_rate_hz=100.0)

        assert result.result_.orientation_resolved is True
        assert result.result_.unresolved_reason is None
        assert result.result_.phase is None

    def test_single_walking_bout(self):
        """Test algorithm on a single walking bout from lab data."""
        single_test = LabExampleDataset(reference_system="INDIP", reference_para_level="wb").get_subset(
            cohort="HA", participant_id="001", test="Test11", trial="Trial1"
        )

        reference_wbs = single_test.reference_parameters_.wb_list
        if len(reference_wbs) == 0:
            pytest.skip("No reference walking bouts available.")

        start = reference_wbs.iloc[0]["start"]
        end = reference_wbs.iloc[0]["end"]

        imu_data = single_test.data.get("LowerBack")
        imu_data = imu_data.reset_index(drop=True)
        wb_data = imu_data.loc[start:end]

        result = ReorientationMethodDM(correction_mode="full").detect_correct(
            wb_data, sampling_rate_hz=single_test.sampling_rate_hz
        )

        # Check that result_ has all required attributes
        assert hasattr(result.result_, "family")
        assert hasattr(result.result_, "where_grav")
        assert hasattr(result.result_, "where_grav_points")
        assert hasattr(result.result_, "phase")
        assert hasattr(result.result_, "correction_applied")
        assert hasattr(result.result_, "correction_action")
        assert hasattr(result.result_, "orientation_resolved")
        assert hasattr(result.result_, "unresolved_reason")
        assert hasattr(result.result_, "correction_rotations")
        assert len(result.result_.correction_rotations) == 2
        assert hasattr(result.result_, "data_corrected")

        # Check corrected data has same shape as input
        assert result.result_.data_corrected.shape == wb_data.shape


class TestReorientationMethodDMRegression:
    """Regression tests to ensure algorithm outputs remain stable."""

    @pytest.mark.parametrize("datapoint", LabExampleDataset(reference_system="INDIP", reference_para_level="wb"))
    def test_example_lab_data_full_correction_mode(self, datapoint, snapshot):
        """Test full correction mode on all lab data walking bouts."""
        data = datapoint.data_ss
        ref_walk_bouts = datapoint.reference_parameters_.wb_list

        if len(ref_walk_bouts) == 0:
            pytest.skip("No reference parameters available.")

        iterator = GsIterator()
        reorientation = ReorientationMethodDM(correction_mode="full")

        families = []
        corrections_applied = []
        correction_actions = []

        for (gs, wb_data), result in iterator.iterate(data, ref_walk_bouts):
            reorientation_result = reorientation.detect_correct(wb_data, sampling_rate_hz=datapoint.sampling_rate_hz)
            families.append(reorientation_result.result_.family)
            corrections_applied.append(reorientation_result.result_.correction_applied)
            correction_actions.append(reorientation_result.result_.correction_action)

        # Create summary dataframe for snapshot
        results_df = pd.DataFrame(
            {
                "family": families,
                "correction_applied": corrections_applied,
                "correction_action": correction_actions,
            }
        )

        snapshot.assert_match(results_df, str(tuple(datapoint.group_label)))

    @pytest.mark.parametrize("datapoint", LabExampleDataset(reference_system="INDIP", reference_para_level="wb"))
    def test_example_lab_data_trust_gravity_correction_mode(self, datapoint, snapshot):
        """Test trust-gravity mode on all lab data walking bouts."""
        data = datapoint.data_ss
        ref_walk_bouts = datapoint.reference_parameters_.wb_list

        if len(ref_walk_bouts) == 0:
            pytest.skip("No reference parameters available.")

        iterator = GsIterator()
        reorientation = ReorientationMethodDM(correction_mode="trust_gravity")

        families = []
        corrections_applied = []
        correction_actions = []

        for (gs, wb_data), result in iterator.iterate(data, ref_walk_bouts):
            reorientation_result = reorientation.detect_correct(wb_data, sampling_rate_hz=datapoint.sampling_rate_hz)
            families.append(reorientation_result.result_.family)
            corrections_applied.append(reorientation_result.result_.correction_applied)
            correction_actions.append(reorientation_result.result_.correction_action)

        # Create summary dataframe for snapshot
        results_df = pd.DataFrame(
            {
                "family": families,
                "correction_applied": corrections_applied,
                "correction_action": correction_actions,
            }
        )

        snapshot.assert_match(results_df, str(tuple(datapoint.group_label)))


class TestReorientationEmulationPipeline:
    def test_example_lab_data_creates_prediction_per_orientation_and_wb(self):
        single_test = LabExampleDataset(reference_system="INDIP", reference_para_level="wb").get_subset(
            cohort="HA",
            participant_id="001",
            test="Test11",
            trial="Trial1",
        )

        result = ReorientationEmulationPipeline(ReorientationMethodDM(correction_mode="full")).run(single_test)

        assert result.predictions_.index.names == ["wb_id"]
        assert result.predictions_.columns.to_list() == ["label", "prediction"]
        assert len(result.predictions_) == len(single_test.reference_parameters_.wb_list) * len(REORIENTATION_LABELS)

        first_wb_id = result.predictions_.index.get_level_values("wb_id")[0]
        first_wb_predictions = result.predictions_per_wb_[first_wb_id]
        assert first_wb_predictions.columns.to_list() == ["label", "prediction"]
        assert first_wb_predictions["label"].to_list() == list(REORIENTATION_LABELS)

    def test_trust_gravity_emulation_includes_uncorrectable_pa_flip(self):
        single_test = LabExampleDataset(reference_system="INDIP", reference_para_level="wb").get_subset(
            cohort="HA",
            participant_id="001",
            test="Test11",
            trial="Trial1",
        )

        result = ReorientationEmulationPipeline(ReorientationMethodDM()).run(single_test)

        assert len(result.predictions_) == len(single_test.reference_parameters_.wb_list) * len(REORIENTATION_LABELS)
        pa_flipped_predictions = result.predictions_.loc[
            result.predictions_["label"] == "pa_flipped__rot_pa_0", "prediction"
        ]
        assert len(pa_flipped_predictions) == len(single_test.reference_parameters_.wb_list)
        assert set(pa_flipped_predictions) == {"identity"}

    def test_reorientation_score_returns_combined_accuracy_and_confusion_matrix(self):
        single_test = LabExampleDataset(reference_system="INDIP", reference_para_level="wb").get_subset(
            cohort="HA",
            participant_id="001",
            test="Test11",
            trial="Trial1",
        )

        result = Evaluation(single_test, scoring=reorientation_score).run(
            ReorientationEmulationPipeline(ReorientationMethodDM(correction_mode="full"))
        )

        agg_results = result.get_aggregated_results_as_df()
        raw_results = result.get_raw_results()

        assert "combined__accuracy" in agg_results.columns
        assert 0 <= agg_results.loc[0, "combined__accuracy"] <= 1
        assert set(raw_results) == {"predictions", "confusion_matrix"}
        assert int(raw_results["confusion_matrix"].to_numpy().sum()) == len(raw_results["predictions"])
