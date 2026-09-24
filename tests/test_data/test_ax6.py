"""Integration checks for the optional AX6 CWA dataset."""

from __future__ import annotations

import pickle
from functools import partial
from os import utime
from shutil import copyfile
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from mobgap.consts import GRAV_MS2, SF_SENSOR_COLS
from mobgap.data import (
    AX6Dataset,
    CwaRecordingInfo,
    get_example_cwa_data_path,
    split_at_frequency,
    split_by_utc_day,
    split_by_utc_hour,
)

cwa_reader_rs = pytest.importorskip("cwa_reader_rs")

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

EXAMPLE_CWA = get_example_cwa_data_path()


def _dataset(*, splitter: pd.DataFrame | Callable[[CwaRecordingInfo], pd.DataFrame] | None = None) -> AX6Dataset:
    return AX6Dataset(
        EXAMPLE_CWA,
        participant_metadata={"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"},
        recording_metadata={"measurement_condition": "free_living"},
        splitter=splitter,
    )


def _split_first_ten_seconds(info: CwaRecordingInfo) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recording": ["first_10_seconds"],
            "start_time": [info.start_time],
            "end_time": [info.start_time + pd.Timedelta(seconds=10)],
            "condition": [info.recording_metadata["measurement_condition"]],
            "sample_rate_hz": [info.cwa_header["sample_rate_hz"]],
        }
    )


def test_reads_real_cwa_as_mobgap_sensor_data() -> None:
    """Load an AX6 fixture with the expected columns, time and units."""
    dataset = _dataset()

    assert dataset.index["recording"].tolist() == ["main"]
    assert dataset.sampling_rate_hz == 100
    data = dataset.data["LowerBack"]
    assert data.columns.tolist() == SF_SENSOR_COLS
    assert len(data) == 72472
    assert data.index[0] == pd.Timestamp("2012-03-27T11:14:57.500Z")
    assert data.iloc[0]["acc_x"] == pytest.approx(-0.21875 * GRAV_MS2)
    assert data.iloc[0]["gyr_x"] == 0


def test_day_split_keeps_the_recording_in_one_utc_day() -> None:
    """A day subset uses the same half-open time window as its index row."""
    dataset = _dataset(splitter=split_by_utc_day)

    assert dataset.index["recording"].tolist() == ["day_1"]
    data = dataset.get_subset(recording="day_1").data_ss
    assert data.index.min() >= dataset.index.iloc[0].start_time
    assert data.index.max() < dataset.index.iloc[0].end_time
    assert len(data) == 72472


@pytest.mark.parametrize(("splitter", "label"), [(split_by_utc_day, "day"), (split_by_utc_hour, "hour")])
def test_calendar_split_cuts_at_utc_midnight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, splitter: Callable[[CwaRecordingInfo], pd.DataFrame], label: str
) -> None:
    """Adjacent day and hour rows use half-open windows at UTC midnight."""
    start = pd.Timestamp("2026-09-24T23:59:59.990Z")
    midnight = pd.Timestamp("2026-09-25T00:00:00Z")
    cuts: list[tuple[float, float]] = []

    monkeypatch.setattr(cwa_reader_rs, "read_header", lambda _path: {"sample_rate_hz": 100.0})
    monkeypatch.setattr(
        cwa_reader_rs,
        "sampling_consistency_report",
        lambda _path: {"start_from_data": start.isoformat(), "end_from_data": midnight.isoformat()},
    )
    monkeypatch.setattr(cwa_reader_rs, "seconds", lambda first, end: (first, end))

    def read_window(_path: str, *, cut: tuple[float, float], **_kwargs: object) -> dict[str, list[float | int]]:
        cuts.append(cut)
        return {
            "timestamp": [start.value // 1000, midnight.value // 1000],
            **{f"acc_{axis}": [0.0, 0.0] for axis in "xyz"},
            **{f"gyro_{axis}": [0.0, 0.0] for axis in "xyz"},
        }

    monkeypatch.setattr(cwa_reader_rs, "read_cwa_file", read_window)
    path = tmp_path / "crosses-midnight.cwa"
    path.touch()
    dataset = AX6Dataset(
        path,
        participant_metadata={"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"},
        recording_metadata={"measurement_condition": "free_living"},
        splitter=splitter,
    )

    assert dataset.index["recording"].tolist() == [f"{label}_1", f"{label}_2"]
    assert dataset.index["end_time"].iloc[0] == midnight
    assert dataset.get_subset(recording=f"{label}_1").data_ss.index.tolist() == [start]
    assert dataset.get_subset(recording=f"{label}_2").data_ss.index.tolist() == [midnight]
    assert cuts == pytest.approx([(0.0, 0.01), (0.01, 0.02)])


def test_dataframe_splitter_selects_a_recording_window() -> None:
    """A fixed table of timed rows can name and select recording windows."""
    start = pd.Timestamp("2012-03-27T11:14:57.500Z")
    splits = pd.DataFrame(
        {
            "test": ["walk_1"],
            "start_time": [start],
            "end_time": [start + pd.Timedelta(seconds=10)],
        }
    )
    dataset = _dataset(splitter=splits)

    assert dataset.clone().index.equals(splits)
    assert len(dataset.get_subset(test="walk_1").data_ss) == 1000


def test_callable_splitter_receives_recording_info_and_survives_serialization() -> None:
    """A named function can use recording metadata in tpcp clones and workers."""
    dataset = _dataset(splitter=_split_first_ten_seconds)
    restored = pickle.loads(pickle.dumps(dataset))
    index = restored.clone().index

    assert index["condition"].tolist() == ["free_living"]
    assert index["sample_rate_hz"].tolist() == [100.0]
    assert len(restored.get_subset(recording="first_10_seconds").data_ss) == 1000


def test_public_frequency_splitter_can_be_configured_with_partial() -> None:
    """A public frequency splitter remains usable after clone and pickle."""
    splitter = partial(split_at_frequency, frequency="30min", label="half_hour")
    dataset = _dataset(splitter=splitter)

    assert dataset.clone().index["recording"].tolist() == ["half_hour_1"]
    assert pickle.loads(pickle.dumps(dataset)).index.equals(dataset.index)


def test_repeated_data_access_reuses_the_last_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The one-entry memory cache avoids a second Rust read."""
    path = tmp_path / "recording.cwa"
    copyfile(EXAMPLE_CWA, path)
    calls = 0
    original_read = cwa_reader_rs.read_cwa_file

    def count_reads(*args: object, **kwargs: object) -> dict:
        nonlocal calls
        calls += 1
        return original_read(*args, **kwargs)

    monkeypatch.setattr(cwa_reader_rs, "read_cwa_file", count_reads)
    dataset = AX6Dataset(
        path,
        participant_metadata={"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"},
        recording_metadata={"measurement_condition": "free_living"},
    )

    assert len(dataset.data_ss) == 72472
    assert len(dataset.data_ss) == 72472
    assert calls == 1

    stat = path.stat()
    utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert len(dataset.data_ss) == 72472
    assert calls == 2
