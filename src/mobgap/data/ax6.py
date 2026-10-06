"""Datasets for raw AX6 CWA recordings."""

from __future__ import annotations

import warnings
from datetime import timezone
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple
from zoneinfo import ZoneInfo

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
_SAMPLING_RATE_DEVIATION_WARNING = (
    "The expected number of samples/the effective sampling rate waries considerable from the expected values. "
    "While this is likely normal and might happen due to clock drift in long recordings, it might be worth "
    "investigating further. Data will be resampled assuming that the recorded start and end dates are correct."
)


def _cwa_reader() -> Any:
    try:
        return import_module("cwa_reader_rs")
    except ModuleNotFoundError as exc:
        if exc.name != "cwa_reader_rs":
            raise
        raise ImportError("AX6 CWA loading requires the mobgap[ax6] or mobgap[weartime] extra.") from exc


class CwaRecordingInfo(NamedTuple):
    """Recording metadata passed to an AX6 splitter."""

    path: Path
    start_time: pd.Timestamp
    last_sample_time: pd.Timestamp
    end_time: pd.Timestamp
    cwa_header: dict[str, Any]
    cwa_timing_report: dict[str, Any]
    recording_metadata: RecordingMetadata
    tz: str = "UTC"


def split_at_frequency(
    info: CwaRecordingInfo, frequency: str, label: str = "window", *, min_duration: pd.Timedelta | None = None
) -> pd.DataFrame:
    """Split a CWA recording at UTC boundaries of a fixed pandas frequency.

    Use ``functools.partial(split_at_frequency, frequency="30min")`` as a
    dataset splitter. Rows use half-open time windows and names such as
    ``window_1``. The optional ``label`` changes that prefix. Set
    ``min_duration`` to omit windows with less recorded time than the threshold.
    """
    start_utc = info.start_time.tz_convert("UTC")
    last_utc = info.last_sample_time.tz_convert("UTC")
    first_boundary = start_utc.floor(frequency) + pd.tseries.frequencies.to_offset(frequency)
    boundaries = [
        info.start_time,
        *pd.date_range(first_boundary, last_utc, freq=frequency).tz_convert(info.start_time.tz),
        info.end_time,
    ]
    splits = pd.DataFrame({"start_time": boundaries[:-1], "end_time": boundaries[1:]})
    if min_duration is not None:
        splits = splits.loc[lambda df_: df_["end_time"] - df_["start_time"] >= min_duration].reset_index(drop=True)
    splits.insert(0, "recording", [f"{label}_{i + 1}" for i in range(len(splits))])
    return splits


def split_by_utc_day(info: CwaRecordingInfo, *, min_duration: pd.Timedelta | None = None) -> pd.DataFrame:
    """Split a CWA recording into half-open UTC calendar days, optionally omitting short days."""
    return split_at_frequency(info, "D", "day", min_duration=min_duration)


def split_by_utc_hour(info: CwaRecordingInfo, *, min_duration: pd.Timedelta | None = None) -> pd.DataFrame:
    """Split a CWA recording into half-open UTC clock hours, optionally omitting short hours."""
    return split_at_frequency(info, "h", "hour", min_duration=min_duration)


def _split_by_local_days(info: CwaRecordingInfo, *, min_duration: pd.Timedelta | None = None) -> pd.DataFrame:
    """Split at calendar midnights in ``info.tz``, including daylight-saving transitions."""
    start_local = info.start_time.tz_convert(info.tz)
    last_local = info.last_sample_time.tz_convert(info.tz)
    first_midnight = start_local.normalize() + pd.DateOffset(days=1)
    midnights = pd.date_range(first_midnight, last_local, freq="D").tz_convert(info.start_time.tz)
    boundaries = [info.start_time, *midnights, info.end_time]
    splits = pd.DataFrame({"start_time": boundaries[:-1], "end_time": boundaries[1:]})
    if min_duration is not None:
        splits = splits.loc[lambda df_: df_["end_time"] - df_["start_time"] >= min_duration].reset_index(drop=True)
    splits.insert(0, "recording", [f"day_{i + 1}" for i in range(len(splits))])
    return splits


@lru_cache(maxsize=128)
def _recording_info(path: Path, _file_identity: tuple[int, int]) -> tuple[dict, dict]:
    reader = _cwa_reader()

    return reader.read_metadata(str(path)), reader.sampling_consistency_report(str(path))


def _clock_timezone(header: dict, tz: str) -> timezone:
    # AX6 clocks retain the offset from their last synchronization; they do not switch at DST changes.
    last_change = header["last_change_time_raw"]
    if last_change is None:
        raise ValueError("Cannot derive the CWA clock offset without `last_change_time_raw`.")
    configured_at = pd.Timestamp(last_change).tz_localize(ZoneInfo(tz))
    return timezone(configured_at.utcoffset())


def _interpreted_metadata(metadata: dict, *, clock_timezone: timezone, output_timezone: str) -> dict:
    """Keep raw reader fields and add timestamps in the requested output timezone."""
    return {
        **metadata,
        **{
            key.removesuffix("_raw"): (
                None if value is None else pd.Timestamp(value).tz_localize(clock_timezone).tz_convert(output_timezone)
            )
            for key, value in metadata.items()
            if key.endswith("_time_raw")
            or key in {"start_from_data_raw", "end_from_data_raw", "start_from_header_raw", "end_from_header_raw"}
        },
    }


def _recording_bounds(
    header: dict, *, tz: str, output_timezone: Literal["utc", "local"]
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    clock_timezone = _clock_timezone(header, tz)
    if output_timezone not in ("utc", "local"):
        raise ValueError("`output_timezone` must be 'utc' or 'local'.")
    output_tz = "UTC" if output_timezone == "utc" else tz
    start = pd.Timestamp(header["start_from_data_raw"]).tz_localize(clock_timezone).tz_convert(output_tz)
    last_sample = pd.Timestamp(header["end_from_data_raw"]).tz_localize(clock_timezone).tz_convert(output_tz)
    end = last_sample + pd.Timedelta(seconds=1 / float(header["sample_rate_hz"]))
    return start, last_sample, end


def _cwa_recording_info(
    path: Path, *, tz: str, output_timezone: Literal["utc", "local"], recording_metadata: RecordingMetadata
) -> CwaRecordingInfo:
    header, timing = _recording_info(path, _file_identity(path))
    if header["start_from_data_raw"] is None or header["end_from_data_raw"] is None:
        raise ValueError(f"The CWA file has no data timestamps: {path}")
    start, last_sample, end = _recording_bounds(header, tz=tz, output_timezone=output_timezone)
    clock_timezone = _clock_timezone(header, tz)
    output_tz = "UTC" if output_timezone == "utc" else tz
    return CwaRecordingInfo(
        path=path,
        start_time=start,
        last_sample_time=last_sample,
        end_time=end,
        cwa_header=_interpreted_metadata(header, clock_timezone=clock_timezone, output_timezone=output_tz),
        cwa_timing_report=_interpreted_metadata(timing, clock_timezone=clock_timezone, output_timezone=output_tz),
        recording_metadata=recording_metadata,
        tz=tz,
    )


def _file_identity(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _load_cwa_data(  # noqa: PLR0917
    path: Path,
    _file_identity: tuple[int, int],
    start_s: float | None,
    end_s: float | None,
    channels: tuple[AdditionalChannel, ...],
    sampling_rate_hz: float,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    clock_timezone: timezone,
    output_timezone: str,
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
        fixed_utc_offset_timezone=clock_timezone,
    )
    frame = raw.tz_convert(output_timezone).rename_axis("time")
    frame = frame.rename(columns={f"gyro_{axis}": f"gyr_{axis}" for axis in "xyz"})
    # The reader omits channels absent from the source file. Keep AX3 acceleration-only recordings loadable without
    # inventing gyroscope values; algorithms that need gyro data must check for their required channels.
    selected_columns = [
        col
        for col in (*SF_SENSOR_COLS, *(col for channel in channels for col in _ADDITIONAL_COLUMNS[channel]))
        if col in frame
    ]
    frame = frame[selected_columns].copy()
    frame[SF_ACC_COLS] *= GRAV_MS2
    return frame.loc[(frame.index >= start_time) & (frame.index < end_time)]


class BaseAX6Dataset(BaseGaitDataset):
    """Read AX6 CWA files, with file discovery and splitting supplied by subclasses.

    Subclasses implement :meth:`_get_file_paths` and :meth:`_get_splits_for_file`.
    They can provide :attr:`_file_path_root` to store paths relative to a dataset root in the index.
    The index and time selection use ``start_time`` and ``end_time`` columns.
    ``tz`` is the timezone of the sensor's last clock synchronization;
    ``output_timezone`` selects UTC or that local timezone for data, index, and metadata timestamps.
    """

    def __init__(
        self,
        *,
        tz: str,
        output_timezone: Literal["utc", "local"] = "utc",
        additional_sensors_enabled: Sequence[AdditionalChannel] = (),
        warn_thres_for_sampling_rate_deviations_hz: float | None = None,
        sensor_name: str = "LowerBack",
        memory: joblib.Memory = joblib.Memory(None),
        groupby_cols: list[str] | str | None = None,
        subset_index: pd.DataFrame | None = None,
    ) -> None:
        self.tz = tz
        self.output_timezone = output_timezone
        self.additional_sensors_enabled = additional_sensors_enabled
        self.warn_thres_for_sampling_rate_deviations_hz = warn_thres_for_sampling_rate_deviations_hz
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
    def _file_path_root(self) -> Path | None:
        """Root for paths stored in the index; ``None`` keeps full paths."""
        return None

    @property
    def _selected_file_path(self) -> Path:
        self.assert_is_single(["file_path"], "_selected_file_path")
        path = Path(self.index.iloc[0].file_path)
        root = self._file_path_root
        return path if root is None else root / path

    def _get_additional_channels(self) -> tuple[AdditionalChannel, ...]:
        channels = tuple(dict.fromkeys(self.additional_sensors_enabled))
        unknown = set(channels) - set(_ADDITIONAL_CHANNELS)
        if unknown:
            raise ValueError(f"Unknown CWA channels: {sorted(unknown)}")
        return channels

    @property
    def cwa_header_(self) -> dict:
        """CWA metadata with raw clock fields and interpreted timestamps in the output timezone."""
        path = self._selected_file_path
        header = _recording_info(path, _file_identity(path))[0]
        return _interpreted_metadata(
            header,
            clock_timezone=_clock_timezone(header, self.tz),
            output_timezone="UTC" if self.output_timezone == "utc" else self.tz,
        )

    @property
    def cwa_timing_report_(self) -> dict:
        """CWA timing report with raw clock fields and interpreted timestamps in the output timezone."""
        path = self._selected_file_path
        header, report = _recording_info(path, _file_identity(path))
        return _interpreted_metadata(
            report,
            clock_timezone=_clock_timezone(header, self.tz),
            output_timezone="UTC" if self.output_timezone == "utc" else self.tz,
        )

    @property
    def sampling_rate_hz(self) -> float:
        """Nominal sampling rate from the selected CWA file header."""
        return float(self.cwa_header_["sample_rate_hz"])

    def create_index(self) -> pd.DataFrame:
        """Combine each file's recording windows into one dataset index."""
        paths = tuple(map(Path, self._get_file_paths()))
        root = self._file_path_root
        splits = []
        for path in paths:
            file_splits = self._get_splits_for_file(path).copy()
            output_tz = "UTC" if self.output_timezone == "utc" else self.tz
            file_splits["start_time"] = pd.to_datetime(file_splits["start_time"], utc=True).dt.tz_convert(output_tz)
            file_splits["end_time"] = pd.to_datetime(file_splits["end_time"], utc=True).dt.tz_convert(output_tz)
            file_splits.insert(0, "file_path", str(path) if root is None else path.relative_to(root).as_posix())
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
        channels = self._get_additional_channels()
        path = self._selected_file_path
        timing = self.cwa_timing_report_
        threshold = self.warn_thres_for_sampling_rate_deviations_hz
        if threshold is not None:
            expected_rate = timing.get("samplingrate_hz_from_header")
            effective_rate = timing.get("samplingrate_hz_from_data")
            if (
                expected_rate is not None
                and effective_rate is not None
                and abs(float(effective_rate) - float(expected_rate)) > threshold
            ):
                warnings.warn(_SAMPLING_RATE_DEVIATION_WARNING, stacklevel=2)
        header = self.cwa_header_
        first_sample, _, full_end = _recording_bounds(header, tz=self.tz, output_timezone=self.output_timezone)
        sampling_rate_hz = self.sampling_rate_hz
        selected = self.index.iloc[0]
        start_time, end_time = selected.start_time, selected.end_time
        start_s = end_s = None
        if start_time != first_sample or end_time != full_end:
            start_s = (start_time - first_sample).total_seconds()
            end_s = (end_time - first_sample).total_seconds()
        return hybrid_cache(self.memory, 1)(_load_cwa_data)(
            path,
            _file_identity(path),
            start_s,
            end_s,
            channels,
            sampling_rate_hz,
            start_time,
            end_time,
            _clock_timezone(header, self.tz),
            "UTC" if self.output_timezone == "utc" else self.tz,
        )


class AX6Dataset(BaseAX6Dataset):
    """Load one or more AX6 CWA files as MobGap sensor-frame data.

    This class can also handle AX3 CWA recordings.

    Install the optional Rust reader with ``pip install mobgap[ax6]``. The
    ``splitter`` determines the index rows. Data is loaded
    only when ``data_ss`` is accessed and is resampled to the nominal header
    sampling rate.

    Parameters
    ----------
    path
        Path to one CWA file, or a sequence of paths. The index includes
        ``file_path`` to identify each recording.
    participant_metadata, recording_metadata
        Metadata required by MobGap pipelines.
    tz
        IANA timezone of the computer that last synchronized the sensor clock. The UTC offset at the header's
        ``last_change_time_raw`` is held fixed throughout the recording. This assumes the last configuration write
        also synchronized the clock; verify that assumption for your configuration software.
    output_timezone
        ``"utc"`` returns UTC timestamps; ``"local"`` converts them to ``tz`` with daylight-saving rules.
    splitter
        A DataFrame with ``start_time`` and ``end_time`` columns plus any
        identifying columns, or a callable returning such a DataFrame.
        A fixed DataFrame supplies the same splits for every file. A callable
        runs separately for each file and receives that file's
        :class:`CwaRecordingInfo`. ``None`` selects each complete recording. Use
        :func:`split_by_utc_day` or :func:`split_by_utc_hour` for UTC calendar
        intervals. Define custom callables at module level so joblib and process
        workers can serialize them.
    additional_sensors_enabled
        Extra CWA channels to return alongside acceleration and gyroscope data.
    warn_thres_for_sampling_rate_deviations_hz
        Warn when the effective and expected sampling rates differ by more than this many Hz. ``None`` disables
        the warning.
    sensor_name
        Key used by ``data`` for this recording.
    memory
        A joblib cache for loaded CWA data. The most recent recording window
        also stays in a one-entry memory cache.
    groupby_cols, subset_index
        Passed to :class:`tpcp.Dataset`.

    Notes
    -----
    Acceleration is returned in m/s², gyroscope data (when recorded) in deg/s and the optional
    magnetometer data in µT. Channels absent from the CWA recording are omitted. Each window is half-open.
    """

    def __init__(
        self,
        path: str | Path | Sequence[str | Path],
        *,
        participant_metadata: ParticipantMetadata,
        recording_metadata: RecordingMetadata,
        tz: str,
        output_timezone: Literal["utc", "local"] = "utc",
        splitter: pd.DataFrame | Callable[[CwaRecordingInfo], pd.DataFrame] | None = None,
        additional_sensors_enabled: Sequence[AdditionalChannel] = (),
        warn_thres_for_sampling_rate_deviations_hz: float | None = None,
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
            tz=tz,
            output_timezone=output_timezone,
            additional_sensors_enabled=additional_sensors_enabled,
            warn_thres_for_sampling_rate_deviations_hz=warn_thres_for_sampling_rate_deviations_hz,
            sensor_name=sensor_name,
            memory=memory,
            groupby_cols=groupby_cols,
            subset_index=subset_index,
        )

    def _get_file_paths(self) -> Sequence[Path]:
        return (Path(self.path),) if isinstance(self.path, (str, Path)) else tuple(map(Path, self.path))

    def _get_splits_for_file(self, path: Path) -> pd.DataFrame:
        info = _cwa_recording_info(
            path,
            tz=self.tz,
            output_timezone=self.output_timezone,
            recording_metadata=self.recording_metadata,
        )
        if self.splitter is None:
            return pd.DataFrame({"recording": ["main"], "start_time": [info.start_time], "end_time": [info.end_time]})
        if isinstance(self.splitter, pd.DataFrame):
            return self.splitter.copy()
        return self.splitter(info)
