from datetime import time

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from mobgap.weartime import WearTimeModelConfig
from mobgap.weartime.utils import clip_intervals_to_waking_hours
from mobgap.weartime.utils.ml_feature_extraction import (
    known_window_mask,
    labels_from_interval_centers,
    window_start_end,
)
from mobgap.weartime.utils.windows_to_weartime import (
    overlapping_window_predictions_to_sample_labels,
    remove_isolated_short_periods_from_intervals,
    remove_short_wear_bouts_by_ratio_from_intervals,
)


def test_training_labels_use_union_of_nested_and_duplicate_wear_regions():
    reference = _intervals([(20, 40), (0, 100), (20, 30), (0, 10)])
    reference["label"] = pd.Categorical(["wear"] * 4, categories=["wear", "uncertain"])

    assert_array_equal(labels_from_interval_centers(np.array([-1, 0, 25, 50, 99, 100]), reference), [0, 1, 1, 1, 1, 0])


@pytest.mark.parametrize("extend_tail, expected", [(False, [1, 1, 0]), (True, [1, 1, 1])])
def test_window_voting_only_extends_wear_into_uncovered_tail_when_requested(extend_tail, expected):
    labels, votes = overlapping_window_predictions_to_sample_labels([1], 3, 2, 2, extend_tail=extend_tail)

    assert_array_equal(labels, expected)
    assert_array_equal(votes[-1], [0, 0])


def _intervals(intervals: list[tuple[int, int]], index_name: str = "wt_id") -> pd.DataFrame:
    return pd.DataFrame(intervals, columns=["start", "end"]).rename_axis(index_name).astype("int64")


def test_clip_intervals_to_waking_hours_drops_empty_boundary_intervals():
    clipped = clip_intervals_to_waking_hours(
        _intervals([(120, 120)]),
        sampling_rate_hz=1.0,
        waking_hours=(time(0, 1), time(0, 2)),
    )

    assert_frame_equal(clipped, _intervals([]))


def test_clip_intervals_to_waking_hours_uses_datetime_index_when_available():
    data = pd.DataFrame(index=pd.date_range("2026-01-01 00:02:00", periods=240, freq="s", tz="UTC"))

    clipped = clip_intervals_to_waking_hours(
        _intervals([(0, 239)]),
        data=data,
        sampling_rate_hz=1.0,
        waking_hours=(time(0, 1), time(0, 3)),
    )

    assert_frame_equal(clipped, _intervals([(0, 60)]))


@pytest.mark.parametrize(
    "index",
    [pd.RangeIndex(240), pd.date_range("2026-01-01 00:02:00", periods=240, freq="s")],
)
def test_clip_intervals_without_localized_timestamps_warns_about_midnight_and_dst(index):
    data = pd.DataFrame(index=index)

    with pytest.warns(UserWarning, match="assum.*midnight.*DST"):
        clipped = clip_intervals_to_waking_hours(
            _intervals([(0, 239)]), data=data, sampling_rate_hz=1.0, waking_hours=(time(0, 1), time(0, 3))
        )

    assert_frame_equal(clipped, _intervals([(60, 180)]))


@pytest.mark.parametrize("index", [pd.RangeIndex(3), pd.date_range("2026-01-01", periods=3, freq="s", tz="UTC")])
def test_waking_clock_boundaries_between_samples_round_up(index):
    data = pd.DataFrame(index=index)

    clipped = clip_intervals_to_waking_hours(
        _intervals([(0, 3)]),
        data=data,
        sampling_rate_hz=1.0,
        waking_hours=(time(0, 0, 0, 500_000), time(0, 0, 1, 500_000)),
    )

    assert_frame_equal(clipped, _intervals([(1, 2)]))


def test_untimed_waking_boundary_exactly_on_high_rate_sample():
    data = pd.DataFrame(index=pd.RangeIndex(62_802))

    clipped = clip_intervals_to_waking_hours(
        _intervals([(0, len(data))]),
        data=data,
        sampling_rate_hz=1000.0,
        waking_hours=(time(0, 1, 2, 800_000), time(0, 1, 2, 801_000)),
    )

    assert_frame_equal(clipped, _intervals([(62_800, 62_801)]))


@pytest.mark.parametrize("index", [pd.RangeIndex(9), pd.date_range("2026-01-01", periods=9, freq="15s", tz="UTC")])
def test_waking_boundary_at_fractional_rate_matches_localized_samples(index):
    data = pd.DataFrame(index=index)

    clipped = clip_intervals_to_waking_hours(
        _intervals([(0, len(data))]),
        data=data,
        sampling_rate_hz=1 / 15,
        waking_hours=(time(0, 1), time(0, 2)),
    )

    assert_frame_equal(clipped, _intervals([(4, 8)]))


def test_untimed_waking_boundary_preserves_a_rate_just_above_one_hertz():
    data = pd.DataFrame(index=pd.RangeIndex(4))

    clipped = clip_intervals_to_waking_hours(
        _intervals([(0, len(data))]),
        data=data,
        sampling_rate_hz=1.0000001,
        waking_hours=(time(0, 0, 1), time(0, 0, 2)),
    )

    assert_frame_equal(clipped, _intervals([(2, 3)]))


def test_clip_intervals_with_localized_timestamps_rejects_multiple_local_days():
    data = pd.DataFrame(index=pd.date_range("2026-03-29 23:59:00", periods=180, freq="s", tz="Europe/London"))

    with pytest.raises(ValueError, match="cross midnight"):
        clip_intervals_to_waking_hours(
            _intervals([(0, 179)]), data=data, sampling_rate_hz=1.0, waking_hours=(time(0), time(1))
        )


@pytest.mark.parametrize("day", ["2026-03-29", "2026-10-25"])
def test_clip_intervals_to_waking_hours_uses_local_clock_time_on_dst_days(day):
    data = pd.DataFrame(index=pd.date_range(f"{day} 06:00", periods=181, freq="min", tz="Europe/Berlin"))

    clipped = clip_intervals_to_waking_hours(
        _intervals([(0, 180)]),
        data=data,
        sampling_rate_hz=1 / 60,
        waking_hours=(time(7), time(8)),
    )

    assert_frame_equal(clipped, _intervals([(60, 120)]))


def test_midnight_end_covers_the_whole_local_day_across_dst():
    start = pd.Timestamp("2026-03-29", tz="Europe/London")
    end = start + pd.DateOffset(days=1)
    data = pd.DataFrame(index=pd.date_range(start, end, freq="min", inclusive="left"))

    clipped = clip_intervals_to_waking_hours(
        _intervals([(0, len(data))]),
        data=data,
        sampling_rate_hz=1 / 60,
        waking_hours=(time(0), time(0)),
    )

    assert_frame_equal(clipped, _intervals([(0, 23 * 60)]))


@pytest.mark.parametrize(
    "day, waking_hours, expected",
    [
        ("2026-03-29", (time(2, 30), time(4)), (60, 120)),
        ("2026-10-25", (time(2), time(2, 30)), (60, 150)),
    ],
)
def test_clip_intervals_to_waking_hours_resolves_transition_hour(day, waking_hours, expected):
    data = pd.DataFrame(index=pd.date_range(f"{day} 01:00", f"{day} 04:00", freq="min", tz="Europe/Berlin"))

    clipped = clip_intervals_to_waking_hours(
        _intervals([(0, len(data) - 1)]),
        data=data,
        sampling_rate_hz=1 / 60,
        waking_hours=waking_hours,
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
        waking_hours=(time(2, 15), time(3)),
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
        waking_hours=(time(7), time(8)),
    )

    assert_frame_equal(clipped, _intervals([(60, 120)]))


class TestRemoveIsolatedShortPeriods:
    def test_removes_short_interior_wear_before_merging_nonwear_gaps(self):
        result = remove_isolated_short_periods_from_intervals(
            np.array([[0, 3], [5, 7]]),
            data_length=10,
            min_period_s=3,
            sampling_rate_hz=1,
        )

        assert_array_equal(result, np.array([[0, 3]]))

    def test_merges_short_interior_nonwear_gaps(self):
        result = remove_isolated_short_periods_from_intervals(
            np.array([[0, 3], [5, 8]]),
            data_length=8,
            min_period_s=3,
            sampling_rate_hz=1,
        )

        assert_array_equal(result, np.array([[0, 8]]))


class TestRemoveShortWearBoutsByRatio:
    def test_removes_short_wear_bout_with_low_surrounding_nonwear_ratio(self):
        result = remove_short_wear_bouts_by_ratio_from_intervals(
            np.array([[5, 7]]),
            data_length=12,
            max_wear_bout_duration_min=3 / 60,
            min_wear_to_nonwear_ratio=0.3,
            sampling_rate_hz=1,
        )

        assert_array_equal(result, np.empty((0, 2), dtype=np.int64))

    def test_keeps_short_wear_bout_with_sufficient_surrounding_nonwear_ratio(self):
        result = remove_short_wear_bouts_by_ratio_from_intervals(
            np.array([[2, 4]]),
            data_length=6,
            max_wear_bout_duration_min=3 / 60,
            min_wear_to_nonwear_ratio=0.3,
            sampling_rate_hz=1,
        )

        assert_array_equal(result, np.array([[2, 4]]))

    def test_keeps_long_wear_bout_regardless_of_surrounding_nonwear_ratio(self):
        result = remove_short_wear_bouts_by_ratio_from_intervals(
            np.array([[5, 9]]),
            data_length=14,
            max_wear_bout_duration_min=3 / 60,
            min_wear_to_nonwear_ratio=0.3,
            sampling_rate_hz=1,
        )

        assert_array_equal(result, np.array([[5, 9]]))


@pytest.mark.parametrize(
    ("uncertain", "expected_starts", "expected_labels"),
    [
        ([], [0, 10, 20, 30, 40], [1, 1, 1, 0, 0]),
        ([(35, 60)], [0, 10], [1, 1]),
        ([(20, 30)], [0, 30, 40], [1, 0, 0]),
        ([(10, 40)], [40], [0]),
        ([(0, 60)], [], []),
    ],
)
def test_uncertainty_excludes_intersecting_windows_without_shifting_labels(
    uncertain: list[tuple[int, int]], expected_starts: list[int], expected_labels: list[int]
) -> None:
    """Remove unknown regions while keeping known windows on the original recording grid."""
    reference = pd.concat(
        [_intervals([(0, 40)]).assign(label="wear"), _intervals(uncertain).assign(label="uncertain")],
        ignore_index=True,
    )
    reference["label"] = pd.Categorical(reference["label"], categories=["wear", "uncertain"])
    windows = window_start_end(60, 20, 10)
    selected = windows[known_window_mask(windows[:, 0], windows[:, 1], reference)]
    assert_array_equal(selected[:, 0], expected_starts)
    assert_array_equal(labels_from_interval_centers(selected[:, 0] + 10, reference), expected_labels)


def test_fractional_sampling_rate_labels_the_rounded_window_center() -> None:
    """A 5-second window at 99.9 Hz has 500 samples, with its midpoint at sample 250."""
    config = WearTimeModelConfig(99.9, 5)
    window_samples, step_samples = config.window_parameters(0)
    windows = window_start_end(500, window_samples, step_samples)
    reference = _intervals([(250, 500)]).assign(label=pd.Categorical(["wear"], categories=["wear", "uncertain"]))
    assert_array_equal(windows, [[0, 500]])
    assert_array_equal(labels_from_interval_centers(windows[:, 0] + window_samples // 2, reference), [1])
