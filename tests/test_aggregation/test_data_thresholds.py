import numpy as np
import pandas as pd
import pytest

from mobgap import PROJECT_ROOT
from mobgap.aggregation import apply_thresholds, get_mobilised_dmo_thresholds


class TestDataThresholds:
    def test_snapshot(self, snapshot):
        input_data = pd.read_csv(
            PROJECT_ROOT / "example_data//original_results/mobilised_aggregator//aggregation_test_input.csv"
        )
        thresholds = get_mobilised_dmo_thresholds()

        flags = apply_thresholds(input_data, thresholds, cohort="HA", height_m=1.7, measurement_condition="free_living")

        # We dropna to remove columns with NaN values to still use the old snapshot
        flags = flags.dropna(axis=1, how="all")

        snapshot.assert_match(flags, "flags")

    def test_heights(self):
        input_data = pd.read_csv(
            PROJECT_ROOT / "example_data//original_results/mobilised_aggregator//aggregation_test_input.csv"
        )
        thresholds = get_mobilised_dmo_thresholds()

        # We naivly test the height by setting to a really large value which should result in all SL values being True
        flags = apply_thresholds(
            input_data, thresholds, cohort="HA", height_m=1000, measurement_condition="free_living"
        )

        assert (flags["stride_length_m"] == True).all()

        # We naivly test the height by setting to a really small value which should result in all SL values being False
        flags = apply_thresholds(input_data, thresholds, cohort="HA", height_m=0.1, measurement_condition="free_living")

        assert (flags["stride_length_m"].all() == False).all()

    def test_missing_required_columns(self):
        # Load input data without all required columns
        input_data = pd.DataFrame({"cadence_spm": [100, 110, 120]})
        thresholds = get_mobilised_dmo_thresholds()

        # Test function behavior when input data is missing required columns
        with pytest.raises(ValueError, match="Input data is missing required columns: .*"):
            apply_thresholds(input_data, thresholds, cohort="HA", height_m=1.7, measurement_condition="free_living")

    def test_invalid_input_data(self):
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
            apply_thresholds(input_data, thresholds, cohort="HA", height_m=1.7, measurement_condition="free_living")

    def test_large_input_data_performance(self):
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
        flags = apply_thresholds(input_data, thresholds, cohort="HA", height_m=1.7, measurement_condition="free_living")
        # Assert some properties of the output if needed
        assert flags.shape[0] == num_rows
        assert flags.shape[1] == input_data.shape[1]  # Ensure the number of columns remains the same
