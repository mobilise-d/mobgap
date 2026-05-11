import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS
from mobgap.data import LabExampleDataset
from mobgap.pipeline import GsIterator
from mobgap.utils.conversions import to_body_frame
from mobgap.re_orientation import ReorientationMethodDM


class TestMetaReorientationMethodDM(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = ReorientationMethodDM

    @pytest.fixture
    def after_action_instance(self):
        algo = self.ALGORITHM_CLASS()
        algo.detect_correct(pd.DataFrame(np.zeros((1000, 6)), columns=BF_SENSOR_COLS))
        return algo


class TestReorientationMethodDM:
    """Tests for ReorientationMethodDM."""

    def test_correctly_oriented_data_family_1(self):
        """Test that correctly oriented data (Family 1) is left unchanged by conservative method."""
        data = pd.DataFrame({
            'acc_is': np.ones(1000) * 9.8,
            'acc_ml': np.random.randn(1000) * 0.5,
            'acc_pa': np.random.randn(1000) * 0.5,
            'gyr_is': np.random.randn(1000) * 0.1,
            'gyr_ml': np.random.randn(1000) * 0.1,
            'gyr_pa': np.random.randn(1000) * 0.1,
        })

        result = ReorientationMethodDM(method='conservative').detect_correct(data)

        assert result.result_.family == 1
        assert result.result_.where_grav == 'is'
        assert result.result_.where_grav_points == 'up'

    def test_family_2_misorientation(self):
        """Test Family 2 (IS pointing down) is corrected."""
        data = pd.DataFrame({
            'acc_is': np.ones(1000) * -9.8,
            'acc_ml': np.random.randn(1000) * 0.5,
            'acc_pa': np.random.randn(1000) * 0.5,
            'gyr_is': np.random.randn(1000) * 0.1,
            'gyr_ml': np.random.randn(1000) * 0.1,
            'gyr_pa': np.random.randn(1000) * 0.1,
        })

        result = ReorientationMethodDM(method='full').detect_correct(data)

        assert result.result_.family == 2
        assert result.result_.where_grav == 'is'
        assert result.result_.where_grav_points == 'down'
        assert result.result_.correction_applied is True
        assert 'flipped IS' in result.result_.correction_action

    def test_family_3_misorientation(self):
        """Test Family 3 (gravity in ML pointing up) is corrected."""
        data = pd.DataFrame({
            'acc_is': np.random.randn(1000) * 0.5,
            'acc_ml': np.ones(1000) * 9.8,
            'acc_pa': np.random.randn(1000) * 0.5,
            'gyr_is': np.random.randn(1000) * 0.1,
            'gyr_ml': np.random.randn(1000) * 0.1,
            'gyr_pa': np.random.randn(1000) * 0.1,
        })

        result = ReorientationMethodDM(method='full').detect_correct(data)

        assert result.result_.family == 3
        assert result.result_.where_grav == 'ml'
        assert result.result_.where_grav_points == 'up'
        assert result.result_.correction_applied is True
        assert 'swapped IS-ML' in result.result_.correction_action

    def test_family_4_misorientation(self):
        """Test Family 4 (gravity in ML pointing down) is corrected."""
        data = pd.DataFrame({
            'acc_is': np.random.randn(1000) * 0.5,
            'acc_ml': np.ones(1000) * -9.8,
            'acc_pa': np.random.randn(1000) * 0.5,
            'gyr_is': np.random.randn(1000) * 0.1,
            'gyr_ml': np.random.randn(1000) * 0.1,
            'gyr_pa': np.random.randn(1000) * 0.1,
        })

        result = ReorientationMethodDM(method='full').detect_correct(data)

        assert result.result_.family == 4
        assert result.result_.where_grav == 'ml'
        assert result.result_.where_grav_points == 'down'
        assert result.result_.correction_applied is True
        assert 'swapped IS-ML' in result.result_.correction_action
        assert 'flipped IS' in result.result_.correction_action

    def test_no_gravity_detected(self):
        """Test that data without clear gravity signal returns None family."""
        data = pd.DataFrame({
            'acc_is': np.random.randn(1000) * 2.0,
            'acc_ml': np.random.randn(1000) * 2.0,
            'acc_pa': np.random.randn(1000) * 2.0,
            'gyr_is': np.random.randn(1000) * 0.1,
            'gyr_ml': np.random.randn(1000) * 0.1,
            'gyr_pa': np.random.randn(1000) * 0.1,
        })

        result = ReorientationMethodDM(method='full').detect_correct(data)

        assert result.result_.family is None
        assert result.result_.where_grav is None
        assert result.result_.correction_applied is False
        assert result.result_.correction_action == 'none'
        # Verify data unchanged
        assert_frame_equal(result.result_.data_corrected, data)

    def test_full_vs_conservative_method_family_1(self):
        """Test that full and conservative methods differ for Family 1."""
        single_test = (
            LabExampleDataset(reference_system="INDIP", reference_para_level="wb")
            .get_subset(cohort="HA", participant_id="001", test="Test11", trial="Trial1")
        )

        reference_wbs = single_test.reference_parameters_.wb_list
        start = reference_wbs.iloc[2]["start"]
        end = reference_wbs.iloc[2]["end"]

        imu_data = to_body_frame(single_test.data.get("LowerBack"))
        imu_data = imu_data.reset_index(drop=True)
        wb_data = imu_data.loc[start:end]

        result_full = ReorientationMethodDM(method='full').detect_correct(wb_data)
        result_conservative = ReorientationMethodDM(method='conservative').detect_correct(wb_data)

        # Both should detect same family
        assert result_full.result_.family == result_conservative.result_.family

        # If Family 1, conservative might skip Stage 3
        if result_full.result_.family == 1:
            assert len(result_conservative.result_.correction_action) <= len(result_full.result_.correction_action)

    def test_invalid_method_parameter(self):
        """Test that invalid method parameter raises ValueError."""
        with pytest.raises(ValueError, match="method must be 'full' or 'conservative'"):
            ReorientationMethodDM(method='invalid').detect_correct(
                pd.DataFrame(np.zeros((1000, 6)), columns=BF_SENSOR_COLS)
            )

    def test_single_walking_bout(self):
        """Test algorithm on a single walking bout from lab data."""
        single_test = (
            LabExampleDataset(reference_system="INDIP", reference_para_level="wb")
            .get_subset(cohort="HA", participant_id="001", test="Test11", trial="Trial1")
        )

        reference_wbs = single_test.reference_parameters_.wb_list
        if len(reference_wbs) == 0:
            pytest.skip("No reference walking bouts available.")

        start = reference_wbs.iloc[0]["start"]
        end = reference_wbs.iloc[0]["end"]

        imu_data = to_body_frame(single_test.data.get("LowerBack"))
        imu_data = imu_data.reset_index(drop=True)
        wb_data = imu_data.loc[start:end]

        result = ReorientationMethodDM(method='full').detect_correct(wb_data)

        # Check that result_ has all required attributes
        assert hasattr(result.result_, 'family')
        assert hasattr(result.result_, 'where_grav')
        assert hasattr(result.result_, 'where_grav_points')
        assert hasattr(result.result_, 'phase')
        assert hasattr(result.result_, 'correction_applied')
        assert hasattr(result.result_, 'correction_action')
        assert hasattr(result.result_, 'data_corrected')

        # Check corrected data has same shape as input
        assert result.result_.data_corrected.shape == wb_data.shape


class TestReorientationMethodDMRegression:
    """Regression tests to ensure algorithm outputs remain stable."""

    @pytest.mark.parametrize(
        "datapoint",
        LabExampleDataset(reference_system="INDIP", reference_para_level="wb")
    )
    def test_example_lab_data_full_method(self, datapoint, snapshot):
        """Test full method on all lab data walking bouts."""
        data = to_body_frame(datapoint.data_ss)
        ref_walk_bouts = datapoint.reference_parameters_.wb_list

        if len(ref_walk_bouts) == 0:
            pytest.skip("No reference parameters available.")

        iterator = GsIterator()
        reorientation = ReorientationMethodDM(method='full')

        families = []
        corrections_applied = []
        correction_actions = []

        for (gs, wb_data), result in iterator.iterate(data, ref_walk_bouts):
            reorientation_result = reorientation.detect_correct(wb_data)
            families.append(reorientation_result.result_.family)
            corrections_applied.append(reorientation_result.result_.correction_applied)
            correction_actions.append(reorientation_result.result_.correction_action)

        # Create summary dataframe for snapshot
        results_df = pd.DataFrame({
            'family': families,
            'correction_applied': corrections_applied,
            'correction_action': correction_actions,
        })

        snapshot.assert_match(results_df, str(tuple(datapoint.group_label)))

    @pytest.mark.parametrize(
        "datapoint",
        LabExampleDataset(reference_system="INDIP", reference_para_level="wb")
    )
    def test_example_lab_data_conservative_method(self, datapoint, snapshot):
        """Test conservative method on all lab data walking bouts."""
        data = to_body_frame(datapoint.data_ss)
        ref_walk_bouts = datapoint.reference_parameters_.wb_list

        if len(ref_walk_bouts) == 0:
            pytest.skip("No reference parameters available.")

        iterator = GsIterator()
        reorientation = ReorientationMethodDM(method='conservative')

        families = []
        corrections_applied = []
        correction_actions = []

        for (gs, wb_data), result in iterator.iterate(data, ref_walk_bouts):
            reorientation_result = reorientation.detect_correct(wb_data)
            families.append(reorientation_result.result_.family)
            corrections_applied.append(reorientation_result.result_.correction_applied)
            correction_actions.append(reorientation_result.result_.correction_action)

        # Create summary dataframe for snapshot
        results_df = pd.DataFrame({
            'family': families,
            'correction_applied': corrections_applied,
            'correction_action': correction_actions,
        })

        snapshot.assert_match(results_df, str(tuple(datapoint.group_label)))