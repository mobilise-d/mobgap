import numpy as np
import pandas as pd
import pytest

from gaitlink import PACKAGE_ROOT
from gaitlink.aggregation import apply_thresholds, get_mobilised_dmo_thresholds


class TestDataThresholds:
    def test_snapshot(self, snapshot):
        input_data = pd.read_csv(
            PACKAGE_ROOT.parent / "example_data//original_results//mobilised_aggregator//aggregation_test_input.csv"
        )
        thresholds = get_mobilised_dmo_thresholds()

        flags = apply_thresholds(input_data, thresholds, cohort="HA", height_m=1.7)

        snapshot.assert_match(flags, "flags")

    def test_thresholds_for_different_cohorts(self, snapshot):
        input_data = pd.read_csv(
            PACKAGE_ROOT.parent / "example_data//original_results//mobilised_aggregator//aggregation_test_input.csv"
        )
        thresholds = get_mobilised_dmo_thresholds()

        # Test for CHF cohort
        flags_chf = apply_thresholds(input_data, thresholds, cohort="CHF", height_m=1.7)
        snapshot.assert_match(flags_chf, "flags_chf")

        # Test for COPD cohort
        flags_copd = apply_thresholds(input_data, thresholds, cohort="COPD", height_m=1.7)
        snapshot.assert_match(flags_copd, "flags_copd")

        # Test for MS cohort
        flags_ms = apply_thresholds(input_data, thresholds, cohort="MS", height_m=1.7)
        snapshot.assert_match(flags_ms, "flags_ms")

    def test_thresholds_for_different_conditions(self, snapshot):
        input_data = pd.read_csv(
            PACKAGE_ROOT.parent / "example_data//original_results//mobilised_aggregator//aggregation_test_input.csv"
        )
        thresholds = get_mobilised_dmo_thresholds()

        # Test for free living condition
        flags_free_living = apply_thresholds(input_data, thresholds, cohort="HA", height_m=1.7, condition="free_living")
        snapshot.assert_match(flags_free_living, "flags_free_living")

        # Test for laboratory condition
        flags_laboratory = apply_thresholds(input_data, thresholds, cohort="HA", height_m=1.7, condition="laboratory")
        snapshot.assert_match(flags_laboratory, "flags_laboratory")

    def test_thresholds_for_different_heights(self, snapshot):
        input_data = pd.read_csv(
            PACKAGE_ROOT.parent / "example_data//original_results//mobilised_aggregator//aggregation_test_input.csv"
        )
        thresholds = get_mobilised_dmo_thresholds()

        # Test for height 1.7 meters
        flags_height_1_7 = apply_thresholds(input_data, thresholds, cohort="HA", height_m=1.7)
        snapshot.assert_match(flags_height_1_7, "flags_height_1_7")

        # Test for height 1.5 meters
        flags_height_1_5 = apply_thresholds(input_data, thresholds, cohort="HA", height_m=1.5)
        snapshot.assert_match(flags_height_1_5, "flags_height_1_5")

    def test_missing_required_columns(snapshot):
        # Load input data without all required columns
        input_data = pd.DataFrame({"cadence_spm": [100, 110, 120]})
        thresholds = get_mobilised_dmo_thresholds()

        # Test function behavior when input data is missing required columns
        with pytest.raises(ValueError, match="Input data is missing required columns: .*"):
            apply_thresholds(input_data, thresholds, cohort="HA", height_m=1.7)

    def test_invalid_input_data(self, snapshot):
        # Load input data with valid numeric values
        input_data = pd.DataFrame(
            {
                "cadence_spm": [100, 110, 120],  # Valid numeric values
                "walking_speed_": [1.2, 1.5, np.nan],
                "stride_length_m": [2.3, 2.8, 3.1],
                "stride_duration_s": [1.2, 1.5, np.nan],
            }
        )
        thresholds = get_mobilised_dmo_thresholds()

        # Test function behavior with invalid input data
        with pytest.raises(ValueError):
            apply_thresholds(input_data, thresholds, cohort="HA", height_m=1.7)

    def test_large_input_data_performance(self, snapshot):
        # Generate large input data for performance testing
        num_rows = 1000000  # 1 million rows
        input_data = pd.DataFrame(
            {
                "cadence_spm": np.random.randint(80, 120, size=num_rows),
                "walking_speed_mps": np.random.uniform(1.0, 1.5, size=num_rows),
                "stride_length_m": np.random.uniform(2.0, 3.0, size=num_rows),
                "stride_duration_s": np.random.uniform(1.0, 2.0, size=num_rows),
            }
        )
        thresholds = get_mobilised_dmo_thresholds()

        # Test function performance with large input data
        flags = apply_thresholds(input_data, thresholds, cohort="HA", height_m=1.7)
        # Assert some properties of the output if needed
        assert flags.shape[0] == num_rows
        assert flags.shape[1] == input_data.shape[1]  # Ensure the number of columns remains the same
