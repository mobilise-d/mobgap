"""Datasets for raw AX6 CWA recordings."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import joblib
import pandas as pd
from tpcp.caching import hybrid_cache

from mobgap.consts import GRAV_MS2, SF_ACC_COLS, SF_SENSOR_COLS
from mobgap.data.base import IMU_DATA_DTYPE, BaseGaitDataset, ParticipantMetadata, RecordingMetadata

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

AdditionalChannel = Literal["battery", "light", "magnetometer", "temperature"]
_ADDITIONAL_CHANNELS = ("battery", "light", "magnetometer", "temperature")
_ADDITIONAL_COLUMNS = {
    "battery": ("battery",),
    "light": ("light",),
    "magnetometer": ("mag_x", "mag_y", "mag_z"),
    "temperature": ("temperature",),
}


def _cwa_reader() -> Any:
    try:
        return import_module("cwa_reader_rs")
    except ModuleNotFoundError as exc:
        if exc.name != "cwa_reader_rs":
            raise
        raise ImportError("AX6 CWA loading requires Python 3.10 or newer and the mobgap[ax6] extra.") from exc


class CwaRecordingInfo(NamedTuple):
    """Recording metadata passed to an AX6 splitter."""

    path: Path
    start_time: pd.Timestamp
    last_sample_time: pd.Timestamp
    end_time: pd.Timestamp
    cwa_header: dict[str, Any]
    cwa_timing_report: dict[str, Any]
    recording_metadata: RecordingMetadata


def split_at_frequency(info: CwaRecordingInfo, frequency: str, label: str = "window") -> pd.DataFrame:
    """Split a CWA recording at UTC boundaries of a fixed pandas frequency.

    Use ``functools.partial(split_at_frequency, frequency="30min")`` as a
    dataset splitter. Rows use half-open time windows and names such as
    ``window_1``. The optional ``label`` changes that prefix.
    """
    first_boundary = info.start_time.floor(frequency) + pd.tseries.frequencies.to_offset(frequency)
    boundaries = [info.start_time, *pd.date_range(first_boundary, info.last_sample_time, freq=frequency), info.end_time]
    return pd.DataFrame(
        {
            "recording": [f"{label}_{i + 1}" for i in range(len(boundaries) - 1)],
            "start_time": boundaries[:-1],
            "end_time": boundaries[1:],
        }
    )


def split_by_utc_day(info: CwaRecordingInfo) -> pd.DataFrame:
    """Split a CWA recording into half-open UTC calendar days."""
    return split_at_frequency(info, "D", "day")


def split_by_utc_hour(info: CwaRecordingInfo) -> pd.DataFrame:
    """Split a CWA recording into half-open UTC clock hours."""
    return split_at_frequency(info, "h", "hour")


@lru_cache(maxsize=128)
def _recording_info(path: Path, _file_identity: tuple[int, int]) -> tuple[dict, dict]:
    reader = _cwa_reader()

    return reader.read_header(str(path)), reader.sampling_consistency_report(str(path))


def _file_identity(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _load_cwa_data(
    path: Path,
    _file_identity: tuple[int, int],
    start_s: float | None,
    end_s: float | None,
    channels: tuple[AdditionalChannel, ...],
    sampling_rate_hz: float,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    reader = _cwa_reader()
    cut = None if start_s is None else reader.seconds(start_s, end_s)
    raw = reader.read_cwa_file(
        str(path),
        cut=cut,
        include_magnetometer="magnetometer" in channels,
        include_temperature="temperature" in channels,
        include_light="light" in channels,
        include_battery="battery" in channels,
        resample_hz=sampling_rate_hz,
        resample_method="cubic",
    )
    frame = pd.DataFrame(raw)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("timestamp"), unit="us", utc=True), name="time")
    frame = frame.rename(columns={f"gyro_{axis}": f"gyr_{axis}" for axis in "xyz"})
    selected_columns = [*SF_SENSOR_COLS, *(col for channel in channels for col in _ADDITIONAL_COLUMNS[channel])]
    frame = frame[selected_columns].copy()
    frame[SF_ACC_COLS] *= GRAV_MS2
    return frame.loc[(frame.index >= start_time) & (frame.index < end_time)]


class BaseAX6Dataset(BaseGaitDataset):
    """Read AX6 CWA files, with file discovery and splitting supplied by subclasses.

    Subclasses implement :meth:`_get_file_paths` and :meth:`_get_splits_for_file`.
    The latter returns rows with ``start_time`` and ``end_time`` columns.
    """

    def __init__(
        self,
        *,
        additional_sensors_enabled: Sequence[AdditionalChannel] = (),
        sensor_name: str = "LowerBack",
        memory: joblib.Memory = joblib.Memory(None),
        groupby_cols: list[str] | str | None = None,
        subset_index: pd.DataFrame | None = None,
    ) -> None:
        self.additional_sensors_enabled = additional_sensors_enabled
        self.sensor_name = sensor_name
        self.memory = memory
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def _get_file_paths(self) -> Sequence[Path]:
        """Return the CWA files represented by this dataset."""
        raise NotImplementedError

    def _get_splits_for_file(self, path: Path) -> pd.DataFrame:
        """Return the recording windows for one CWA file."""
        raise NotImplementedError

    @property
    def _selected_file_path(self) -> Path:
        self.assert_is_single(["file_path"], "_selected_file_path")
        return Path(self.index.iloc[0].file_path)

    @property
    def cwa_header_(self) -> dict:
        """Metadata from the selected CWA file header."""
        path = self._selected_file_path
        return dict(_recording_info(path, _file_identity(path))[0])

    @property
    def cwa_timing_report_(self) -> dict:
        """Timing derived from the selected CWA file's data packets."""
        path = self._selected_file_path
        return dict(_recording_info(path, _file_identity(path))[1])

    @property
    def sampling_rate_hz(self) -> float:
        """Nominal sampling rate from the selected CWA file header."""
        return float(self.cwa_header_["sample_rate_hz"])

    def create_index(self) -> pd.DataFrame:
        """Combine each file's recording windows into one dataset index."""
        paths = tuple(map(Path, self._get_file_paths()))
        splits = []
        for path in paths:
            file_splits = self._get_splits_for_file(path).copy()
            file_splits.insert(0, "file_path", str(path))
            splits.append(file_splits)
        return pd.concat(splits, ignore_index=True)

    @property
    def data(self) -> IMU_DATA_DTYPE:
        """The selected recording as a sensor-name-to-data mapping."""
        return {self.sensor_name: self.data_ss}

    @property
    def data_ss(self) -> pd.DataFrame:
        """The selected recording window in the MobGap sensor frame."""
        self.assert_is_single(None, "data_ss")
        channels = tuple(dict.fromkeys(self.additional_sensors_enabled))
        unknown = set(channels) - set(_ADDITIONAL_CHANNELS)
        if unknown:
            raise ValueError(f"Unknown CWA channels: {sorted(unknown)}")

        row = self.index.iloc[0]
        path = self._selected_file_path
        timing = self.cwa_timing_report_
        first_sample = pd.Timestamp(timing["start_from_data"]).tz_convert("UTC")
        sampling_rate_hz = self.sampling_rate_hz
        full_end = pd.Timestamp(timing["end_from_data"]).tz_convert("UTC") + pd.Timedelta(seconds=1 / sampling_rate_hz)
        start_s = end_s = None
        if row.start_time != first_sample or row.end_time != full_end:
            start_s = (row.start_time - first_sample).total_seconds()
            end_s = (row.end_time - first_sample).total_seconds()
        return hybrid_cache(self.memory, 1)(_load_cwa_data)(
            path, _file_identity(path), start_s, end_s, channels, sampling_rate_hz, row.start_time, row.end_time
        )


class AX6Dataset(BaseAX6Dataset):
    """Load one or more AX6 CWA files as MobGap sensor-frame data.

    This class can also handle AX3 CWA recordings.

    Install the optional Rust reader with ``pip install mobgap[ax6]`` on Python
    3.10 or newer. The ``splitter`` determines the index rows. Data is loaded
    only when ``data_ss`` is accessed and is resampled to the nominal header
    sampling rate.

    Parameters
    ----------
    path
        Path to one CWA file, or a sequence of paths. The index includes
        ``file_path`` to identify each recording.
    participant_metadata, recording_metadata
        Metadata required by MobGap pipelines.
    splitter
        A DataFrame with ``start_time`` and ``end_time`` columns plus any
        identifying columns, or a callable that receives
        :class:`CwaRecordingInfo` and returns such a DataFrame.
        The same DataFrame is applied to every file. ``None`` selects each
        complete recording. Use
        :func:`split_by_utc_day` or :func:`split_by_utc_hour` for UTC calendar
        intervals. Define custom callables at module level so joblib and process
        workers can serialize them.
    additional_sensors_enabled
        Extra CWA channels to return alongside acceleration and gyroscope data.
    sensor_name
        Key used by ``data`` for this recording.
    memory
        A joblib cache for loaded CWA data. The most recent recording window
        also stays in a one-entry memory cache.
    groupby_cols, subset_index
        Passed to :class:`tpcp.Dataset`.

    Notes
    -----
    Acceleration is returned in m/s², gyroscope data in deg/s and the optional
    magnetometer data in µT. The time index is UTC and each day is half-open.
    """

    def __init__(
        self,
        path: str | Path | Sequence[str | Path],
        *,
        participant_metadata: ParticipantMetadata,
        recording_metadata: RecordingMetadata,
        splitter: pd.DataFrame | Callable[[CwaRecordingInfo], pd.DataFrame] | None = None,
        additional_sensors_enabled: Sequence[AdditionalChannel] = (),
        sensor_name: str = "LowerBack",
        memory: joblib.Memory = joblib.Memory(None),
        groupby_cols: list[str] | str | None = None,
        subset_index: pd.DataFrame | None = None,
    ) -> None:
        self.path = path
        self.participant_metadata = participant_metadata
        self.recording_metadata = recording_metadata
        self.splitter = splitter
        super().__init__(
            additional_sensors_enabled=additional_sensors_enabled,
            sensor_name=sensor_name,
            memory=memory,
            groupby_cols=groupby_cols,
            subset_index=subset_index,
        )

    def _get_file_paths(self) -> Sequence[Path]:
        return (Path(self.path),) if isinstance(self.path, (str, Path)) else tuple(map(Path, self.path))

    def _get_splits_for_file(self, path: Path) -> pd.DataFrame:
        header, timing = _recording_info(path, _file_identity(path))
        if timing["start_from_data"] is None or timing["end_from_data"] is None:
            raise ValueError(f"The CWA file has no data timestamps: {path}")

        start = pd.Timestamp(timing["start_from_data"]).tz_convert("UTC")
        last_sample = pd.Timestamp(timing["end_from_data"]).tz_convert("UTC")
        end = last_sample + pd.Timedelta(seconds=1 / float(header["sample_rate_hz"]))
        info = CwaRecordingInfo(
            path=path,
            start_time=start,
            last_sample_time=last_sample,
            end_time=end,
            cwa_header=dict(header),
            cwa_timing_report=timing,
            recording_metadata=self.recording_metadata,
        )
        if self.splitter is None:
            return pd.DataFrame({"recording": ["main"], "start_time": [start], "end_time": [end]})
        if isinstance(self.splitter, pd.DataFrame):
            return self.splitter.copy()
        return self.splitter(info)
