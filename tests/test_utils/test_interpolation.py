import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal
from pandas._testing import assert_series_equal

from mobgap.utils.interpolation import interval_mean, naive_sec_paras_to_regions


class TestIntervalMean:
    def test_no_samples(self):
        measurement_samples = np.array([])
        measurements = np.array([])
        interval_start_ends = np.array([[0, 2], [1, 3]])
        interval_values = interval_mean(measurement_samples, measurements, interval_start_ends)
        assert np.all(np.isnan(interval_values))
        assert len(interval_values) == len(interval_start_ends)

    def test_no_intervals(self):
        measurement_samples = np.array([0])
        measurements = np.array([1])
        interval_start_ends = np.array([])
        interval_values = interval_mean(measurement_samples, measurements, interval_start_ends)
        assert len(interval_values) == 0

    def test_measurements_match_intervals(self):
        measurement_samples = np.array([0.5, 1.5, 2.5])
        measurements = np.array([1, 2, 3])
        interval_start_ends = pd.DataFrame({"start": [0, 1, 2], "end": [1, 2, 3]}).to_numpy()
        interval_values = interval_mean(measurement_samples, measurements, interval_start_ends)
        assert_equal(interval_values, measurements)
        assert len(interval_values) == len(interval_start_ends)

    def test_no_value_in_single_interval(self):
        measurement_samples = np.array([0.5, 2.5])
        measurements = np.array([1, 3])
        interval_start_ends = np.array([[0, 1], [1, 2], [2, 3]])
        interval_values = interval_mean(measurement_samples, measurements, interval_start_ends)

        expected = np.array([1, np.nan, 3])

        assert_equal(interval_values, expected)
        assert len(interval_values) == len(interval_start_ends)

    def test_multiple_values_in_single_interval(self):
        measurement_samples = np.array([0.5, 1.5, 2.5])
        measurements = np.array([1, 2, 3])
        interval_start_ends = np.array([[0, 2], [1, 3]])
        interval_values = interval_mean(measurement_samples, measurements, interval_start_ends)

        expected = np.array([1.5, 2.5])

        assert_equal(interval_values, expected)
        assert len(interval_values) == len(interval_start_ends)

    def test_measurements_on_start_included(self):
        measurement_samples = np.array([0, 1, 2])
        measurements = np.array([1, 2, 3])
        interval_start_ends = np.array([[0, 0.5], [1, 1.5], [2, 2.5]])
        interval_values = interval_mean(measurement_samples, measurements, interval_start_ends)

        expected = np.array([1, 2, 3])

        assert_equal(interval_values, expected)
        assert len(interval_values) == len(interval_start_ends)

    def test_measurements_on_end_included(self):
        measurement_samples = np.array([1, 2, 3])
        measurements = np.array([1, 2, 3])
        interval_start_ends = np.array([[0.5, 1], [1.5, 2], [2.5, 3]])
        interval_values = interval_mean(measurement_samples, measurements, interval_start_ends)

        expected = np.array([1, 2, 3])

        assert_equal(interval_values, expected)
        assert len(interval_values) == len(interval_start_ends)

    def test_nan_measurement_is_ignored(self):
        measurement_samples = np.array([0, 1, 2])
        measurements = np.array([1, np.nan, 3])
        interval_start_ends = np.array([[0, 1], [1, 2]])
        interval_values = interval_mean(measurement_samples, measurements, interval_start_ends)

        expected = np.array([1, 3])

        assert_equal(interval_values, expected)
        assert len(interval_values) == len(interval_start_ends)


class TestNaiveSecParasToRegions:
    def test_empty_inputs(self):
        region_list = pd.DataFrame(columns=["start", "end"])
        sec_paras = pd.DataFrame(columns=["sec_center_samples", "value"]).set_index("sec_center_samples")
        result = naive_sec_paras_to_regions(region_list, sec_paras, sampling_rate_hz=1)
        assert result.empty
        assert result.columns.tolist() == ["start", "end", "value"]

    def test_empty_region_list(self):
        region_list = pd.DataFrame(columns=["start", "end"])
        sec_paras = pd.DataFrame({"sec_center_samples": [0.5], "value": [1]}, index=[0]).set_index("sec_center_samples")
        result = naive_sec_paras_to_regions(region_list, sec_paras, sampling_rate_hz=1)
        assert result.empty
        assert result.columns.tolist() == ["start", "end", "value"]

    def test_empty_sec_paras(self):
        region_list = pd.DataFrame({"start": [0], "end": [1]}, index=[0])
        sec_paras = pd.DataFrame(columns=["sec_center_samples", "value"]).set_index("sec_center_samples")
        result = naive_sec_paras_to_regions(region_list, sec_paras, sampling_rate_hz=1)
        assert result["value"].isna().all()
        assert len(result) == len(region_list)

    @pytest.mark.parametrize("n_values", [1, 2, 3, 10])
    def test_exact_second_interpolation(self, n_values):
        values_per_sec = 10
        starts = np.arange(n_values) * values_per_sec
        ends = starts + values_per_sec
        centers = starts + values_per_sec / 2
        region_list = pd.DataFrame({"start": starts, "end": ends})
        sec_paras = pd.DataFrame({"sec_center_samples": centers, "value": starts.astype(float)}).set_index(
            "sec_center_samples"
        )
        result = naive_sec_paras_to_regions(region_list, sec_paras, sampling_rate_hz=values_per_sec)
        assert not result.empty
        assert_series_equal(result["value"], sec_paras["value"], check_index=False)

    @pytest.mark.parametrize("n_values", [1, 2, 3, 10])
    def test_half_sec_interpolation(self, n_values):
        values_per_sec = 10
        starts = np.arange(n_values) * values_per_sec
        ends = starts + values_per_sec / 2
        centers = starts + values_per_sec / 2
        region_list = pd.DataFrame({"start": starts, "end": ends})
        sec_paras = pd.DataFrame({"sec_center_samples": centers, "value": starts.astype(float)}).set_index(
            "sec_center_samples"
        )
        result = naive_sec_paras_to_regions(region_list, sec_paras, sampling_rate_hz=values_per_sec)
        assert not result.empty
        assert_series_equal(result["value"], sec_paras["value"], check_index=False)

    @pytest.mark.parametrize("n_values", [1, 2, 3, 10])
    def test_shifted_half_sec_interpolation(self, n_values):
        values_per_sec = 10
        sec_starts = np.arange(n_values + 1) * values_per_sec
        # We can only have one less
        starts = sec_starts[:-1] + values_per_sec / 2
        ends = starts + values_per_sec
        centers = sec_starts + values_per_sec / 2
        region_list = pd.DataFrame({"start": starts, "end": ends})
        sec_paras = pd.DataFrame({"sec_center_samples": centers, "value": centers.astype(float)}).set_index(
            "sec_center_samples"
        )
        result = naive_sec_paras_to_regions(region_list, sec_paras, sampling_rate_hz=values_per_sec)
        assert not result.empty
        assert_series_equal(result["value"], sec_paras["value"].iloc[:-1] + values_per_sec / 2, check_index=False)
