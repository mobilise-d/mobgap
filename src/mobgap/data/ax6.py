"""A dataset for one raw AX6 CWA recording."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import joblib
import pandas as pd
from tpcp.caching import hybrid_cache

from mobgap.consts import GRAV_MS2, SF_ACC_COLS, SF_SENSOR_COLS
from mobgap.data.base import IMU_DATA_DTYPE, BaseGaitDataset, ParticipantMetadata, RecordingMetadata

if TYPE_CHECKING:
    from collections.abc import Sequence

AdditionalChannel = Literal["battery", "light", "magnetometer", "temperature"]
_ADDITIONAL_CHANNELS = ("battery", "light", "magnetometer", "temperature")
_ADDITIONAL_COLUMNS = {
    "battery": ("battery",),
    "light": ("light",),
    "magnetometer": ("mag_x", "mag_y", "mag_z"),
    "temperature": ("temperature",),
}


@lru_cache(maxsize=128)
def _recording_info(path: Path, _file_identity: tuple[int, int]) -> tuple[dict, dict]:
    reader = import_module("cwa_reader_rs")

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
    reader = import_module("cwa_reader_rs")
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


class SingleRecordingDataset(BaseGaitDataset):
    """Load one AX6 CWA file as MobGap sensor-frame data.

    The optional ``ax6`` dependency provides the Rust CWA reader. Each index row
    selects the whole recording or one UTC calendar day. Data is loaded only when
    ``data_ss`` is accessed and is resampled to the nominal header sampling rate.

    Parameters
    ----------
    path
        Path to one CWA file.
    participant_metadata, recording_metadata
        Metadata required by MobGap pipelines.
    split_into_days
        Make one index row per UTC calendar day instead of one per recording.
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
        path: str | Path,
        *,
        participant_metadata: ParticipantMetadata,
        recording_metadata: RecordingMetadata,
        split_into_days: bool = False,
        additional_sensors_enabled: Sequence[AdditionalChannel] = (),
        sensor_name: str = "LowerBack",
        memory: joblib.Memory = joblib.Memory(None),
        groupby_cols: list[str] | str | None = None,
        subset_index: pd.DataFrame | None = None,
    ) -> None:
        self.path = path
        self.participant_metadata = participant_metadata
        self.recording_metadata = recording_metadata
        self.split_into_days = split_into_days
        self.additional_sensors_enabled = additional_sensors_enabled
        self.sensor_name = sensor_name
        self.memory = memory
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def cwa_header_(self) -> dict:
        """Metadata from the CWA file header."""
        path = Path(self.path)
        return dict(_recording_info(path, _file_identity(path))[0])

    @property
    def cwa_timing_report_(self) -> dict:
        """Timing derived from the CWA data packets."""
        path = Path(self.path)
        return dict(_recording_info(path, _file_identity(path))[1])

    @property
    def sampling_rate_hz(self) -> float:
        """Nominal sampling rate from the CWA header."""
        return float(self.cwa_header_["sample_rate_hz"])

    def create_index(self) -> pd.DataFrame:
        """Index the recording, optionally splitting it at UTC midnight."""
        timing = self.cwa_timing_report_
        if timing["start_from_data"] is None or timing["end_from_data"] is None:
            raise ValueError(f"The CWA file has no data timestamps: {self.path}")

        start = pd.Timestamp(timing["start_from_data"]).tz_convert("UTC")
        last_sample = pd.Timestamp(timing["end_from_data"]).tz_convert("UTC")
        end = last_sample + pd.Timedelta(seconds=1 / self.sampling_rate_hz)
        boundaries = [start]
        if self.split_into_days:
            boundaries.extend(pd.date_range(start.normalize() + pd.Timedelta(days=1), last_sample, freq="D"))
        boundaries.append(end)

        return pd.DataFrame(
            {
                "recording": [f"day_{i + 1}" for i in range(len(boundaries) - 1)] if self.split_into_days else ["main"],
                "start_time": boundaries[:-1],
                "end_time": boundaries[1:],
            }
        )

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
        start_s = end_s = None
        if self.split_into_days:
            first_sample = pd.Timestamp(self.cwa_timing_report_["start_from_data"]).tz_convert("UTC")
            start_s = (row.start_time - first_sample).total_seconds()
            end_s = (row.end_time - first_sample).total_seconds()
        path = Path(self.path)
        return hybrid_cache(self.memory, 1)(_load_cwa_data)(
            path, _file_identity(path), start_s, end_s, channels, self.sampling_rate_hz, row.start_time, row.end_time
        )
