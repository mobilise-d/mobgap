import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mobgap import PROJECT_ROOT
from mobgap.aggregation import apply_thresholds, get_mobilised_dmo_thresholds


class TestDataThresholds:
    @pytest.fixture
    def simple_thresholds(self):
        dmos = ["cadence_spm", "walking_speed_mps", "stride_length_m", "stride_duration_s"]
        return pd.DataFrame(
            np.tile([1.0, 2.0], (4, 1)),
            index=pd.MultiIndex.from_product([dmos, ["HA"]], names=["dmo", "cohort"]),
            columns=pd.MultiIndex.from_product([["free_living"], ["min", "max"]]),
        )

    def test_inclusive_boundaries_and_dataframe_metadata(self, simple_thresholds):
        index = pd.MultiIndex.from_tuples(
            [("b", 3), ("a", 1), ("b", 1), ("a", 2), ("c", 0)], names=["participant", "wb_id"]
        )
        columns = pd.Index(list(reversed(simple_thresholds.index.get_level_values("dmo"))), name="measure")
        values = [np.nextafter(1.0, -np.inf), 1.0, 2.0, np.nextafter(2.0, np.inf), np.nan]
        data = pd.DataFrame({column: values for column in columns}, index=index).reindex(columns=columns)
        data["unchecked"] = "metadata"

        result = apply_thresholds(
            data, simple_thresholds, cohort="HA", height_m=None, measurement_condition="free_living", check_against=()
        )

        expected = pd.DataFrame({column: [False, True, True, False, False] for column in columns}, index=index).reindex(
            columns=columns
        )
        expected["unchecked"] = np.nan
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["float32", "int64", "Float64", "Int64"])
    def test_numeric_dtypes_and_threshold_precision(self, dtype, simple_thresholds):
        simple_thresholds.loc[:, ("free_living", "min")] = 1.00000001
        simple_thresholds.loc[:, ("free_living", "max")] = 1.00000002
        columns = simple_thresholds.index.get_level_values("dmo")
        data = pd.DataFrame(1, index=[0, 1], columns=columns, dtype=dtype)
        if dtype in ("Float64", "Int64"):
            data.iloc[1, :] = pd.NA

        result = apply_thresholds(
            data, simple_thresholds, cohort="HA", height_m=None, measurement_condition="free_living", check_against=()
        )

        nullable = dtype in ("Float64", "Int64")
        expected = pd.DataFrame(
            {column: [False, pd.NA if nullable else False] for column in columns},
            dtype="boolean" if nullable else bool,
        ).reindex(columns=columns)
        assert_frame_equal(result, expected)

    def test_empty_input_dtypes(self, simple_thresholds):
        data = pd.DataFrame(columns=simple_thresholds.index.get_level_values("dmo"), dtype="float64")

        result = apply_thresholds(
            data, simple_thresholds, cohort="HA", height_m=None, measurement_condition="free_living", check_against=()
        )

        assert_frame_equal(result, data)

    def test_duplicate_indices(self, simple_thresholds):
        data = pd.DataFrame(1.0, index=[0, 0], columns=simple_thresholds.index.get_level_values("dmo"))

        with pytest.raises(ValueError, match="Index contains duplicate entries"):
            apply_thresholds(
                data,
                simple_thresholds,
                cohort="HA",
                height_m=None,
                measurement_condition="free_living",
                check_against=(),
            )

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
