import numpy as np
import pandas as pd
from numpy.testing import assert_equal

from gaitlink.utils.interpolation import interval_mean


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
