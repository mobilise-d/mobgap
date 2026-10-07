"""Integration checks for the optional AX6 CWA dataset."""

from __future__ import annotations

import pickle
import warnings
from functools import partial
from os import utime
from shutil import copyfile
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from mobgap.consts import GRAV_MS2, SF_ACC_COLS
from mobgap.data import (
    AX6Dataset,
    BaseAX6Dataset,
    CwaRecordingInfo,
    get_example_cwa_data_path,
    split_at_frequency,
    split_by_local_days,
    split_by_utc_day,
    split_by_utc_hour,
)
from mobgap.data import ax6 as ax6_module

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
        tz="UTC",
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


def _split_named_for_file(info: CwaRecordingInfo) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recording": [info.path.stem],
            "start_time": [info.start_time],
            "end_time": [info.start_time + pd.Timedelta(seconds=10)],
        }
    )


class _DiscoveredFilesDataset(BaseAX6Dataset):
    def __init__(
        self, paths: list[Path], groupby_cols: list[str] | str | None = None, subset_index: pd.DataFrame | None = None
    ) -> None:
        self.paths = paths
        self.participant_metadata = {"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"}
        self.recording_metadata = {"measurement_condition": "free_living"}
        super().__init__(tz="UTC", groupby_cols=groupby_cols, subset_index=subset_index)

    def _get_file_paths(self) -> list[Path]:
        return self.paths

    def _get_splits_for_file(self, path: Path) -> pd.DataFrame:
        start = pd.Timestamp("2012-03-27T11:14:57.500Z")
        return pd.DataFrame(
            {"recording": [path.stem], "start_time": [start], "end_time": [start + pd.Timedelta(seconds=10)]}
        )


def test_reads_real_cwa_as_mobgap_sensor_data() -> None:
    """Load the acceleration-only CWA fixture without fabricating gyro values."""
    dataset = _dataset()

    assert dataset.index["recording"].tolist() == ["main"]
    assert dataset.index["file_path"].tolist() == [str(EXAMPLE_CWA)]
    assert dataset.sampling_rate_hz == 100
    data = dataset.data["LowerBack"]
    assert data.columns.tolist() == SF_ACC_COLS
    assert len(data) == 72472
    assert data.index[0] == pd.Timestamp("2012-03-27T11:14:57.500Z")
    assert data.iloc[0]["acc_x"] == pytest.approx(-0.21875 * GRAV_MS2)


def test_reader_uses_configuration_offset_and_requested_output_timezone() -> None:
    """The sensor clock keeps its configuration offset, while local output uses the named timezone."""
    options = {
        "participant_metadata": {"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"},
        "recording_metadata": {"measurement_condition": "free_living"},
        "tz": "Europe/Berlin",
    }
    utc_dataset = AX6Dataset(EXAMPLE_CWA, output_timezone="utc", **options)
    local_dataset = AX6Dataset(EXAMPLE_CWA, **options)
    utc_data = utc_dataset.data_ss
    local_data = local_dataset.data_ss

    assert utc_data.index[0] == pd.Timestamp("2012-03-27T09:14:57.500Z")
    assert local_data.index[0] == pd.Timestamp("2012-03-27T11:14:57.500+02:00")
    assert local_data.index.tz == pd.Timestamp.now(tz="Europe/Berlin").tz
    assert local_data.iloc[0].equals(utc_data.iloc[0])
    assert local_dataset.index.start_time.iloc[0] == local_data.index[0]
    assert local_dataset.cwa_header_["start_from_data"] == local_data.index[0]
    assert local_dataset.cwa_timing_report_["start_from_data"] == local_data.index[0]
    assert utc_dataset.cwa_header_["start_from_data"] == utc_data.index[0]


def test_utc_clock_needs_no_configuration_timestamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "unconfigured.cwa"
    path.touch()
    header = {
        "sample_rate_hz": 100.0,
        "last_change_time_raw": None,
        "start_from_data_raw": "2026-03-29T10:00:00",
        "end_from_data_raw": "2026-03-29T10:00:00.990",
    }
    monkeypatch.setattr(ax6_module, "_recording_info", lambda *_args: (header, {}))
    dataset = AX6Dataset(
        path,
        tz="UTC",
        participant_metadata={"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"},
        recording_metadata={"measurement_condition": "free_living"},
    )

    assert dataset.index.start_time.iloc[0] == pd.Timestamp("2026-03-29T10:00:00Z")
    assert dataset.cwa_header_["last_change_time"] is None


@pytest.mark.parametrize(
    ("last_change", "start_raw", "end_raw", "day_start", "day_end", "hours"),
    [
        (
            "2026-03-20T12:00:00",
            "2026-03-28T23:00:00",
            "2026-03-30T00:59:59.990",
            "2026-03-28T23:00:00Z",
            "2026-03-29T22:00:00Z",
            23,
        ),
        (
            "2026-10-20T12:00:00",
            "2026-10-24T23:00:00",
            "2026-10-27T00:59:59.990",
            "2026-10-24T22:00:00Z",
            "2026-10-25T23:00:00Z",
            25,
        ),
    ],
)
def test_local_day_split_follows_calendar_at_clock_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    last_change: str,
    start_raw: str,
    end_raw: str,
    day_start: str,
    day_end: str,
    hours: int,
) -> None:
    """A Berlin calendar day can span 23 or 25 elapsed hours at a clock change."""
    path = tmp_path / "spring.cwa"
    path.touch()
    header = {
        "sample_rate_hz": 100.0,
        "last_change_time_raw": last_change,
        "start_from_data_raw": start_raw,
        "end_from_data_raw": end_raw,
    }
    monkeypatch.setattr(ax6_module, "_recording_info", lambda *_args: (header, {}))
    dataset = AX6Dataset(
        path,
        tz="Europe/Berlin",
        participant_metadata={"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"},
        recording_metadata={"measurement_condition": "free_living"},
        splitter=split_by_local_days,
        output_timezone="utc",
    )

    assert dataset.index.start_time.iloc[1] == pd.Timestamp(day_start)
    assert dataset.index.end_time.iloc[1] == pd.Timestamp(day_end)
    assert dataset.index.end_time.iloc[1] - dataset.index.start_time.iloc[1] == pd.Timedelta(hours=hours)


def test_additional_sensors_enabled_parameter_survives_clone() -> None:
    """The shared channel parameter controls optional CWA columns."""
    dataset = AX6Dataset(
        EXAMPLE_CWA,
        participant_metadata={"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"},
        recording_metadata={"measurement_condition": "free_living"},
        tz="UTC",
        additional_sensors_enabled=("temperature", "magnetometer"),
    )

    assert dataset.clone().data_ss.columns.tolist() == [*SF_ACC_COLS, "temperature"]


def test_sampling_rate_deviation_warning_threshold() -> None:
    """The base loader owns the optional warning for CWA clock drift."""
    options = {
        "participant_metadata": {"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"},
        "recording_metadata": {"measurement_condition": "free_living"},
        "tz": "UTC",
    }
    with pytest.warns(UserWarning, match="effective sampling rate waries considerable"):
        AX6Dataset(EXAMPLE_CWA, warn_thres_for_sampling_rate_deviations_hz=0.2, **options).data_ss

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AX6Dataset(EXAMPLE_CWA, warn_thres_for_sampling_rate_deviations_hz=10.0, **options).data_ss
    assert not [warning for warning in caught if "effective sampling rate" in str(warning.message)]


def test_day_split_keeps_the_recording_in_one_utc_day() -> None:
    """A day subset uses the same half-open time window as its index row."""
    dataset = _dataset(splitter=split_by_utc_day)

    assert dataset.index["recording"].tolist() == ["day_1"]
    data = dataset.get_subset(recording="day_1").data_ss
    assert data.index.min() >= dataset.index.iloc[0].start_time
    assert data.index.max() < dataset.index.iloc[0].end_time
    assert len(data) == 72472


@pytest.mark.parametrize(
    ("splitter", "expected_start", "expected_end"),
    [
        (split_by_utc_day, "2026-09-25T00:00:00Z", "2026-09-25T01:30:00Z"),
        (split_by_utc_hour, "2026-09-25T00:00:00Z", "2026-09-25T01:00:00Z"),
    ],
)
def test_calendar_splitter_omits_windows_shorter_than_min_duration(
    splitter: Callable[..., pd.DataFrame], expected_start: str, expected_end: str
) -> None:
    """A partial calendar window shorter than the requested duration is omitted."""
    info = CwaRecordingInfo(
        path=EXAMPLE_CWA,
        start_time=pd.Timestamp("2026-09-24T23:30:00Z"),
        last_sample_time=pd.Timestamp("2026-09-25T01:29:59.990Z"),
        end_time=pd.Timestamp("2026-09-25T01:30:00Z"),
        cwa_header={"sample_rate_hz": 100.0},
        cwa_timing_report={},
        recording_metadata={},
        tz="UTC",
    )

    splits = splitter(info, min_duration=pd.Timedelta(hours=1))

    assert splits.start_time.tolist() == [pd.Timestamp(expected_start)]
    assert splits.end_time.tolist() == [pd.Timestamp(expected_end)]


@pytest.mark.parametrize(("splitter", "label"), [(split_by_utc_day, "day"), (split_by_utc_hour, "hour")])
def test_calendar_split_cuts_at_utc_midnight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, splitter: Callable[[CwaRecordingInfo], pd.DataFrame], label: str
) -> None:
    """Adjacent day and hour rows use half-open windows at UTC midnight."""
    start = pd.Timestamp("2026-09-24T23:59:59.990Z")
    midnight = pd.Timestamp("2026-09-25T00:00:00Z")
    cuts: list[tuple[float, float]] = []

    monkeypatch.setattr(
        cwa_reader_rs,
        "read_metadata",
        lambda _path: {
            "sample_rate_hz": 100.0,
            "last_change_time_raw": "2026-09-24T12:00:00",
            "start_from_data_raw": start.tz_localize(None).isoformat(),
            "end_from_data_raw": midnight.tz_localize(None).isoformat(),
        },
    )
    monkeypatch.setattr(
        cwa_reader_rs,
        "sampling_consistency_report",
        lambda _path: {
            "start_from_data_raw": start.tz_localize(None).isoformat(),
            "end_from_data_raw": midnight.tz_localize(None).isoformat(),
        },
    )
    monkeypatch.setattr(cwa_reader_rs, "seconds", lambda first, end: (first, end))

    def read_window(_path: str, *, cut: tuple[float, float], **_kwargs: object) -> pd.DataFrame:
        cuts.append(cut)
        return pd.DataFrame(
            {
                **{f"acc_{axis}": [0.0, 0.0] for axis in "xyz"},
                **{f"gyro_{axis}": [0.0, 0.0] for axis in "xyz"},
            },
            index=pd.DatetimeIndex([start, midnight], name="timestamp"),
        )

    monkeypatch.setattr(cwa_reader_rs, "read_cwa_file", read_window)
    path = tmp_path / "crosses-midnight.cwa"
    path.touch()
    dataset = AX6Dataset(
        path,
        participant_metadata={"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"},
        recording_metadata={"measurement_condition": "free_living"},
        tz="UTC",
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

    assert dataset.clone().index.drop(columns="file_path").equals(splits)
    assert len(dataset.get_subset(test="walk_1").data_ss) == 1000


def test_fixed_split_table_is_exposed_in_requested_local_timezone() -> None:
    start = pd.Timestamp("2012-03-27T09:14:57.500Z")
    splits = pd.DataFrame(
        {"recording": ["first"], "start_time": [start], "end_time": [start + pd.Timedelta(seconds=10)]}
    )
    dataset = AX6Dataset(
        EXAMPLE_CWA,
        tz="Europe/Berlin",
        output_timezone="local",
        participant_metadata={"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"},
        recording_metadata={"measurement_condition": "free_living"},
        splitter=splits,
    )

    assert dataset.index.start_time.iloc[0] == pd.Timestamp("2012-03-27T11:14:57.500+02:00")
    assert dataset.data_ss.index[0] == dataset.index.start_time.iloc[0]


def test_fixed_split_table_accepts_mixed_dst_offsets() -> None:
    splits = pd.DataFrame(
        {
            "recording": ["before", "after"],
            "start_time": [pd.Timestamp("2026-03-29T01:00:00+01:00"), pd.Timestamp("2026-03-29T03:00:00+02:00")],
            "end_time": [pd.Timestamp("2026-03-29T01:30:00+01:00"), pd.Timestamp("2026-03-29T03:30:00+02:00")],
        }
    )
    dataset = AX6Dataset(
        EXAMPLE_CWA,
        tz="Europe/Berlin",
        output_timezone="local",
        participant_metadata={"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"},
        recording_metadata={"measurement_condition": "free_living"},
        splitter=splits,
    )

    assert dataset.index.start_time.tolist() == [
        pd.Timestamp("2026-03-29T01:00:00+01:00"),
        pd.Timestamp("2026-03-29T03:00:00+02:00"),
    ]
    assert str(dataset.index.start_time.dt.tz) == "Europe/Berlin"


def test_fixed_split_table_rejects_naive_timestamps() -> None:
    splits = pd.DataFrame(
        {
            "recording": ["naive"],
            "start_time": [pd.Timestamp("2012-03-27T11:15:00")],
            "end_time": [pd.Timestamp("2012-03-27T11:15:10")],
        }
    )
    dataset = AX6Dataset(
        EXAMPLE_CWA,
        tz="Europe/Berlin",
        participant_metadata={"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"},
        recording_metadata={"measurement_condition": "free_living"},
        splitter=splits,
    )

    with pytest.raises(TypeError, match="tz-naive"):
        _ = dataset.index


def test_single_file_metadata_is_available_with_multiple_windows() -> None:
    """Header and timing metadata remain available before selecting a window."""
    start = pd.Timestamp("2012-03-27T11:14:57.500Z")
    splits = pd.DataFrame(
        {
            "recording": ["first", "second"],
            "start_time": [start, start + pd.Timedelta(seconds=10)],
            "end_time": [start + pd.Timedelta(seconds=10), start + pd.Timedelta(seconds=20)],
        }
    )
    dataset = _dataset(splitter=splits)

    assert len(dataset.index) == 2
    assert dataset.cwa_header_["sample_rate_hz"] == 100
    assert dataset.cwa_timing_report_["start_from_data_raw"] is not None
    assert dataset.sampling_rate_hz == 100


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


def test_multiple_files_keep_splits_distinct_and_load_the_selected_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A multi-file dataset identifies and reads the selected file and window."""
    paths = [tmp_path / "first.cwa", tmp_path / "second.cwa"]
    for path in paths:
        copyfile(EXAMPLE_CWA, path)

    reads: list[str] = []
    original_read = cwa_reader_rs.read_cwa_file

    def record_path(path: str, **kwargs: object) -> pd.DataFrame:
        reads.append(path)
        return original_read(path, **kwargs)

    monkeypatch.setattr(cwa_reader_rs, "read_cwa_file", record_path)
    dataset = AX6Dataset(
        paths,
        participant_metadata={"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"},
        recording_metadata={"measurement_condition": "free_living"},
        tz="UTC",
        splitter=_split_named_for_file,
    )

    assert dataset.index["file_path"].tolist() == [str(path) for path in paths]
    assert dataset.index["recording"].tolist() == ["first", "second"]
    selected = dataset.get_subset(file_path=str(paths[1]))
    assert len(selected.data_ss) == 1000
    assert reads == [str(paths[1])]
    assert pickle.loads(pickle.dumps(dataset)).clone().index.equals(dataset.index)


def test_fixed_split_table_applies_to_each_file(tmp_path: Path) -> None:
    """The same fixed windows appear once for each CWA file."""
    paths = [tmp_path / "first.cwa", tmp_path / "second.cwa"]
    for path in paths:
        copyfile(EXAMPLE_CWA, path)
    start = pd.Timestamp("2012-03-27T11:14:57.500Z")
    splits = pd.DataFrame(
        {
            "recording": ["first_10s"],
            "start_time": [start],
            "end_time": [start + pd.Timedelta(seconds=10)],
        }
    )

    dataset = AX6Dataset(
        paths,
        participant_metadata={"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"},
        recording_metadata={"measurement_condition": "free_living"},
        tz="UTC",
        splitter=splits,
    )

    assert dataset.index["file_path"].tolist() == [str(path) for path in paths]
    assert dataset.index["recording"].tolist() == ["first_10s", "first_10s"]
    assert len(dataset.get_subset(file_path=str(paths[1])).data_ss) == 1000


def test_base_dataset_can_use_subclass_file_discovery_and_splits(tmp_path: Path) -> None:
    """A subclass can supply files and windows without replacing the CWA loader."""
    paths = [tmp_path / "first.cwa", tmp_path / "second.cwa"]
    for path in paths:
        copyfile(EXAMPLE_CWA, path)

    dataset = _DiscoveredFilesDataset(paths)

    assert dataset.index["recording"].tolist() == ["first", "second"]
    assert len(dataset.get_subset(recording="second").data_ss) == 1000


def test_missing_optional_reader_explains_extra_requirement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional dependency error explains how to install the reader."""
    path = tmp_path / "recording.cwa"
    copyfile(EXAMPLE_CWA, path)

    def missing_reader(_name: str) -> None:
        raise ModuleNotFoundError("No module named 'cwa_reader_rs'", name="cwa_reader_rs")

    monkeypatch.setattr(ax6_module, "import_module", missing_reader)
    dataset = AX6Dataset(
        path,
        participant_metadata={"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"},
        recording_metadata={"measurement_condition": "free_living"},
        tz="UTC",
    )

    with pytest.raises(ImportError, match=r"mobgap\[ax6\]"):
        _ = dataset.index


def test_repeated_data_access_reuses_the_last_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The one-entry memory cache avoids a second Rust read."""
    path = tmp_path / "recording.cwa"
    copyfile(EXAMPLE_CWA, path)
    calls = 0
    original_read = cwa_reader_rs.read_cwa_file

    def count_reads(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return original_read(*args, **kwargs)

    monkeypatch.setattr(cwa_reader_rs, "read_cwa_file", count_reads)
    dataset = AX6Dataset(
        path,
        participant_metadata={"height_m": 1.7, "sensor_height_m": 1.0, "cohort": "HA"},
        recording_metadata={"measurement_condition": "free_living"},
        tz="UTC",
    )

    assert len(dataset.data_ss) == 72472
    assert len(dataset.data_ss) == 72472
    assert calls == 1

    stat = path.stat()
    utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert len(dataset.data_ss) == 72472
    assert calls == 2
