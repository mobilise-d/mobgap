import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from mobgap.weartime.utils import clip_intervals_to_waking_hours
from mobgap.weartime.utils.windows_to_weartime import (
    remove_isolated_short_periods_from_intervals,
    remove_short_wear_bouts_by_ratio_from_intervals,
)


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


@pytest.mark.parametrize("day", ["2026-03-29", "2026-10-25"])
def test_clip_intervals_to_waking_hours_uses_local_clock_time_on_dst_days(day):
    data = pd.DataFrame(index=pd.date_range(f"{day} 06:00", periods=181, freq="min", tz="Europe/Berlin"))

    clipped = clip_intervals_to_waking_hours(
        _intervals([(0, 180)]),
        data=data,
        sampling_rate_hz=1 / 60,
        waking_hours_min=(7 * 60, 8 * 60),
    )

    assert_frame_equal(clipped, _intervals([(60, 120)]))


@pytest.mark.parametrize(
    "day, waking_hours_min, expected",
    [
        ("2026-03-29", (150, 240), (60, 120)),
        ("2026-10-25", (120, 150), (60, 150)),
    ],
)
def test_clip_intervals_to_waking_hours_resolves_transition_hour(day, waking_hours_min, expected):
    data = pd.DataFrame(index=pd.date_range(f"{day} 01:00", f"{day} 04:00", freq="min", tz="Europe/Berlin"))

    clipped = clip_intervals_to_waking_hours(
        _intervals([(0, len(data) - 1)]),
        data=data,
        sampling_rate_hz=1 / 60,
        waking_hours_min=waking_hours_min,
    )

    assert_frame_equal(clipped, _intervals([expected]))


def test_clip_intervals_to_waking_hours_keeps_time_after_half_hour_dst_jump():
    data = pd.DataFrame(
        index=pd.date_range("2026-10-04 01:00", "2026-10-04 03:00", freq="min", tz="Australia/Lord_Howe")
    )

    clipped = clip_intervals_to_waking_hours(
        _intervals([(0, len(data) - 1)]),
        data=data,
        sampling_rate_hz=1 / 60,
        waking_hours_min=(2 * 60 + 15, 3 * 60),
    )

    assert_frame_equal(clipped, _intervals([(60, 90)]))


@pytest.mark.parametrize(
    ("day", "timezone"),
    [("2018-08-12", "America/Santiago"), ("2020-11-01", "America/Havana")],
)
def test_clip_intervals_to_waking_hours_handles_midnight_clock_change(day, timezone):
    data = pd.DataFrame(index=pd.date_range(f"{day} 06:00", periods=181, freq="min", tz=timezone))

    clipped = clip_intervals_to_waking_hours(
        _intervals([(0, 180)]),
        data=data,
        sampling_rate_hz=1 / 60,
        waking_hours_min=(7 * 60, 8 * 60),
    )

    assert_frame_equal(clipped, _intervals([(60, 120)]))


class TestRemoveIsolatedShortPeriods:
    def test_removes_short_interior_wear_before_merging_nonwear_gaps(self):
        result = remove_isolated_short_periods_from_intervals(
            np.array([[0, 3], [5, 7]]),
            data_length=10,
            min_period_sec=3,
            sampling_rate_hz=1,
        )

        assert_array_equal(result, np.array([[0, 3]]))

    def test_merges_short_interior_nonwear_gaps(self):
        result = remove_isolated_short_periods_from_intervals(
            np.array([[0, 3], [5, 8]]),
            data_length=8,
            min_period_sec=3,
            sampling_rate_hz=1,
        )

        assert_array_equal(result, np.array([[0, 8]]))


class TestRemoveShortWearBoutsByRatio:
    def test_removes_short_wear_bout_with_low_surrounding_nonwear_ratio(self):
        result = remove_short_wear_bouts_by_ratio_from_intervals(
            np.array([[5, 7]]),
            data_length=12,
            max_bout_minutes=3 / 60,
            min_ratio=0.3,
            sampling_rate_hz=1,
        )

        assert_array_equal(result, np.empty((0, 2), dtype=np.int64))

    def test_keeps_short_wear_bout_with_sufficient_surrounding_nonwear_ratio(self):
        result = remove_short_wear_bouts_by_ratio_from_intervals(
            np.array([[2, 4]]),
            data_length=6,
            max_bout_minutes=3 / 60,
            min_ratio=0.3,
            sampling_rate_hz=1,
        )

        assert_array_equal(result, np.array([[2, 4]]))

    def test_keeps_long_wear_bout_regardless_of_surrounding_nonwear_ratio(self):
        result = remove_short_wear_bouts_by_ratio_from_intervals(
            np.array([[5, 9]]),
            data_length=14,
            max_bout_minutes=3 / 60,
            min_ratio=0.3,
            sampling_rate_hz=1,
        )

        assert_array_equal(result, np.array([[5, 9]]))
