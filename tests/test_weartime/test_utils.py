import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from mobgap.weartime.utils import clip_intervals_to_waking_hours
from mobgap.weartime.utils.ml_feature_extraction import remove_short_wear_bouts_by_ratio
from mobgap.weartime.utils.windows_to_weartime import remove_isolated_short_periods


def _intervals(intervals: list[tuple[int, int]], index_name: str = "wt_id") -> pd.DataFrame:
    return pd.DataFrame(intervals, columns=["start", "end"]).rename_axis(index_name).astype("int64")


def test_clip_intervals_to_waking_hours_drops_empty_boundary_intervals():
    clipped = clip_intervals_to_waking_hours(
        _intervals([(120, 120)]),
        sampling_rate_hz=1.0,
        waking_hours_min=(1, 2),
    )

    assert_frame_equal(clipped, _intervals([]))


def test_clip_intervals_to_waking_hours_uses_datetime_index_when_available():
    data = pd.DataFrame(index=pd.date_range("2026-01-01 00:02:00", periods=240, freq="s", tz="UTC"))

    clipped = clip_intervals_to_waking_hours(
        _intervals([(0, 239)]),
        data=data,
        sampling_rate_hz=1.0,
        waking_hours_min=(1, 3),
    )

    assert_frame_equal(clipped, _intervals([(0, 60)]))


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
