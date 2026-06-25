import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal

from mobgap.weartime.utils.ml_feature_extraction import remove_short_wear_bouts_by_ratio
from mobgap.weartime.utils.weartime_calc import (
    generate_weartime_list_from_minutes,
    generate_weartime_list_from_samples,
    generate_weartime_list_from_seconds,
)
from mobgap.weartime.utils.windows_to_weartime import remove_isolated_short_periods


def _weartime_df(start_ends: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(start_ends, columns=["start", "end"]).rename_axis(index="wt_id")


class TestGenerateWeartimeList:
    def test_detects_wear_intervals_at_boundaries(self):
        flags = np.array([1, 1, 0, 1, 0, 1, 1])

        result = generate_weartime_list_from_samples(flags)

        assert_frame_equal(result, _weartime_df([(0, 2), (3, 4), (5, 7)]))

    def test_handles_empty_and_all_nonwear_flags(self):
        expected = _weartime_df([]).astype({"start": "int64", "end": "int64"})

        assert_frame_equal(generate_weartime_list_from_samples(np.array([], dtype=int)), expected)
        assert_frame_equal(generate_weartime_list_from_samples(np.zeros(4, dtype=int)), expected)

    def test_scales_second_and_minute_flags_to_samples(self):
        second_flags = np.array([0, 1, 1, 0, 1])
        minute_flags = np.array([1, 0, 1])

        assert_frame_equal(
            generate_weartime_list_from_seconds(second_flags, sampling_rate=20),
            _weartime_df([(20, 60), (80, 100)]),
        )
        assert_frame_equal(
            generate_weartime_list_from_minutes(minute_flags, sampling_rate=20),
            _weartime_df([(0, 1200), (2400, 3600)]),
        )


class TestRemoveIsolatedShortPeriods:
    def test_removes_short_interior_wear_before_merging_nonwear_gaps(self):
        flags = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0])

        result = remove_isolated_short_periods(flags, min_period_sec=3, sampling_rate_hz=1)

        assert_array_equal(result, np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]))

    def test_merges_short_interior_nonwear_gaps(self):
        flags = np.array([1, 1, 1, 0, 0, 1, 1, 1])

        result = remove_isolated_short_periods(flags, min_period_sec=3, sampling_rate_hz=1)

        assert_array_equal(result, np.ones(8, dtype=int))

    def test_keeps_short_boundary_periods(self):
        flags = np.array([0, 0, 1, 1, 1, 0, 0])

        result = remove_isolated_short_periods(flags, min_period_sec=3, sampling_rate_hz=1)

        assert_array_equal(result, flags)


class TestRemoveShortWearBoutsByRatio:
    def test_removes_short_wear_bout_with_low_surrounding_nonwear_ratio(self):
        flags = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])

        result = remove_short_wear_bouts_by_ratio(
            flags,
            max_bout_minutes=3 / 60,
            min_ratio=0.3,
            sampling_rate_hz=1,
        )

        assert_array_equal(result, np.zeros_like(flags))

    def test_keeps_short_wear_bout_with_sufficient_surrounding_nonwear_ratio(self):
        flags = np.array([0, 0, 1, 1, 0, 0])

        result = remove_short_wear_bouts_by_ratio(
            flags,
            max_bout_minutes=3 / 60,
            min_ratio=0.3,
            sampling_rate_hz=1,
        )

        assert_array_equal(result, flags)

    def test_keeps_long_wear_bout_regardless_of_surrounding_nonwear_ratio(self):
        flags = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        result = remove_short_wear_bouts_by_ratio(
            flags,
            max_bout_minutes=3 / 60,
            min_ratio=0.3,
            sampling_rate_hz=1,
        )

        assert_array_equal(result, flags)
