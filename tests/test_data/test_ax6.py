"""Integration checks for the optional AX6 CWA dataset."""

from __future__ import annotations

from os import utime
from pathlib import Path
from shutil import copyfile

import pandas as pd
import pytest

from mobgap.consts import GRAV_MS2, SF_SENSOR_COLS
from mobgap.data import SingleRecordingDataset

cwa_reader_rs = pytest.importorskip("cwa_reader_rs")

EXAMPLE_CWA = Path(__file__).parent / "data" / "ax6" / "example-610-steps.cwa"


def _dataset(*, split_into_days: bool = False) -> SingleRecordingDataset:
    return SingleRecordingDataset(
        EXAMPLE_CWA,
        participant_metadata={"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"},
        recording_metadata={"measurement_condition": "free_living"},
        split_into_days=split_into_days,
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
    dataset = _dataset(split_into_days=True)

    assert dataset.index["recording"].tolist() == ["day_1"]
    data = dataset.get_subset(recording="day_1").data_ss
    assert data.index.min() >= dataset.index.iloc[0].start_time
    assert data.index.max() < dataset.index.iloc[0].end_time
    assert len(data) == 72472


def test_day_split_cuts_at_utc_midnight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Adjacent day rows use half-open reader windows at UTC midnight."""
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
    dataset = SingleRecordingDataset(
        path,
        participant_metadata={"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"},
        recording_metadata={"measurement_condition": "free_living"},
        split_into_days=True,
    )

    assert dataset.index["recording"].tolist() == ["day_1", "day_2"]
    assert dataset.index["end_time"].iloc[0] == midnight
    assert dataset.get_subset(recording="day_1").data_ss.index.tolist() == [start]
    assert dataset.get_subset(recording="day_2").data_ss.index.tolist() == [midnight]
    assert cuts == pytest.approx([(0.0, 0.01), (0.01, 0.02)])


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
    dataset = SingleRecordingDataset(
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
