"""Dataset loader for the SUSTAIN wear-time recordings."""

from __future__ import annotations

import re
import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union

import joblib
import pandas as pd
from tpcp import cf
from tpcp.caching import hybrid_cache

from mobgap.data import ax6 as ax6_module
from mobgap.data.ax6 import AdditionalChannel, BaseAX6Dataset, CwaRecordingInfo, split_by_local_days
from mobgap.utils.array_handling import merge_intervals

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mobgap.data.base import ParticipantMetadata, RecordingMetadata

PathLike = Union[str, Path]
MissingReferenceErrorType = Literal["raise", "warn", "ignore"]
REFERENCE_COLUMNS = ["start", "end", "duration", "start_dt", "end_dt", "duration_s"]
DEFAULT_WARN_THRES_FOR_SAMPLING_RATE_DEVIATIONS_HZ = 0.2
_DEFAULT_DAILY_SPLITTER = partial(split_by_local_days, min_duration=pd.Timedelta(hours=8))


def _is_lowerback_name(name: str) -> bool:
    name = name.lower()
    compact_name = name.replace("_", "").replace("-", "").replace(" ", "")
    tokens = {token for token in re.split(r"[\W_]+", name) if token}
    return "lowerback" in compact_name or "lowback" in compact_name or "lb" in tokens


def _as_utc_timestamp(timestamp: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _reference_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("Europe/London").tz_convert("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def _load_reference_file(reference_path: PathLike) -> pd.DataFrame:
    reference_path = Path(reference_path)
    if not reference_path.exists():
        raise FileNotFoundError(f"Could not find the SUSTAIN wear-time reference file at {reference_path}.")

    reference = pd.read_json(reference_path, lines=True, dtype={"id": "string", "sensor": "string"})
    if reference.empty:
        return pd.DataFrame(columns=["participant_id", "device_off", "device_on", "wear_status"])

    required_columns = {"id", "sensor", "device_off", "device_on", "wear_status"}
    missing_columns = required_columns - set(reference.columns)
    if missing_columns:
        raise ValueError(
            f"The SUSTAIN wear-time reference file is missing the following columns: {sorted(missing_columns)}."
        )

    reference = (
        reference.assign(
            participant_id=lambda df_: df_["id"].astype("string").str.zfill(3),
            is_lowerback=lambda df_: df_["sensor"].map(_is_lowerback_name),
            wear_status=lambda df_: df_["wear_status"].astype("string"),
        )
        .loc[lambda df_: df_["is_lowerback"]]
        .assign(
            device_off=lambda df_: df_["device_off"].map(_reference_timestamp),
            device_on=lambda df_: df_["device_on"].map(_reference_timestamp),
        )
        .drop(columns=["id", "sensor", "is_lowerback"])
    )

    return reference.sort_values(["participant_id", "device_off", "device_on"], ignore_index=True)


def _empty_reference_df(index_name: str, data_index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start": pd.Series(dtype="int64"),
            "end": pd.Series(dtype="int64"),
            "duration": pd.Series(dtype="int64"),
            "start_dt": pd.Series(dtype=data_index.dtype),
            "end_dt": pd.Series(dtype=data_index.dtype),
            "duration_s": pd.Series(dtype="float64"),
        }
    ).rename_axis(index_name)


def _sample_boundary_timestamp(
    data_index: pd.DatetimeIndex, sample_boundary: int, sampling_rate_hz: float
) -> pd.Timestamp:
    if sample_boundary >= len(data_index):
        return data_index[-1] + pd.to_timedelta(1 / sampling_rate_hz, unit="s")
    return data_index[sample_boundary]


def _timestamp_to_sample_boundary(timestamp: Any, data_index: pd.DatetimeIndex, fallback: int) -> int:
    if pd.isna(timestamp):
        return fallback
    if len(data_index) == 0:
        return 0

    timestamp = _as_utc_timestamp(timestamp)
    if timestamp < data_index[0]:
        return 0
    if timestamp > data_index[-1]:
        return len(data_index)

    return int(data_index.get_indexer([timestamp], method="nearest")[0])


def _format_reference_df(
    intervals: pd.DataFrame,
    *,
    data_index: pd.DatetimeIndex,
    sampling_rate_hz: float,
    index_name: str,
) -> pd.DataFrame:
    valid_intervals = intervals.loc[intervals["end"] > intervals["start"], ["start", "end"]]
    intervals = pd.DataFrame(merge_intervals(valid_intervals.to_numpy(dtype="int64")), columns=["start", "end"])
    if intervals.empty:
        return _empty_reference_df(index_name, data_index)

    intervals = intervals.assign(
        duration=lambda df_: df_["end"] - df_["start"],
        start_dt=lambda df_: [
            _sample_boundary_timestamp(data_index, sample_boundary, sampling_rate_hz)
            for sample_boundary in df_["start"]
        ],
        end_dt=lambda df_: [
            _sample_boundary_timestamp(data_index, sample_boundary, sampling_rate_hz) for sample_boundary in df_["end"]
        ],
        duration_s=lambda df_: df_["duration"] / sampling_rate_hz,
    )
    intervals = intervals[REFERENCE_COLUMNS].astype({"start": "int64", "end": "int64", "duration": "int64"})
    intervals.index.name = index_name
    return intervals


def _complement_intervals(intervals: pd.DataFrame, data_length: int) -> pd.DataFrame:
    if data_length == 0:
        return pd.DataFrame(columns=["start", "end"])

    complement: list[tuple[int, int]] = []
    current_start = 0
    for start, end in intervals[["start", "end"]].itertuples(index=False):
        if start > current_start:
            complement.append((current_start, int(start)))
        current_start = max(current_start, int(end))
    if current_start < data_length:
        complement.append((current_start, data_length))
    return pd.DataFrame(complement, columns=["start", "end"])


class SustainWearTimeDataset(BaseAX6Dataset):
    """Dataset for the SUSTAIN wear-time raw CWA recordings.

    The dataset index contains one row per selected time window, including ``file_path``, ``start_time`` and
    ``end_time``. ``file_path`` is relative to ``base_path``, so the index does not depend on where the dataset is
    stored. ``recording_day`` is the UK local date at the start of the window. The raw data is loaded lazily and
    returned in the MobGap sensor frame. Reference intervals use MobGap's half-open sample convention: ``start`` is
    inclusive and ``end`` is exclusive. ``start_dt`` and ``end_dt`` come from snapped sample boundaries.
    Reference timestamps without an offset are interpreted as UK local time.

    Parameters
    ----------
    base_path
        The root folder containing ``weartime_part_a_all`` and ``weartime_part_b``.
    tz
        Timezone of the computer that synchronized the sensor clock. Use ``"Europe/London"``
        for SUSTAIN recordings configured in UK local time.
    output_timezone
        ``"local"`` returns timestamps in ``tz`` (the default); ``"utc"`` returns UTC timestamps.
    additional_sensors_enabled
        Additional CWA channels to append to the core accelerometer and gyroscope data. Supports
        ``"temperature"``, ``"light"``, ``"battery"`` and ``"magnetometer"``.
    missing_reference_error_type
        How to handle missing part A reference rows for a selected recording.
    warn_thres_for_sampling_rate_deviations_hz
        Threshold in Hz used to warn when the effective sampling rate differs from the expected sampling rate. Set to
        ``None`` to disable the warning.
    sensor_name
        Sensor key used by ``data``. Defaults to ``"LowerBack"``.
    splitter
        A DataFrame of windows or a callable receiving one file's :class:`~mobgap.data.CwaRecordingInfo` and
        returning a DataFrame of windows. By default, the dataset selects UK local calendar days with at least
        eight hours of recorded data using :func:`~mobgap.data.split_by_local_days`.
        Set to ``None`` to use each complete recording. Use
        :func:`~mobgap.data.split_by_utc_day` or :func:`~mobgap.data.split_by_utc_hour` for UTC calendar splits.
    memory
        A joblib memory object used to cache CWA data and reference file loading.
    groupby_cols
        Columns to group the data by. See :class:`~tpcp.Dataset` for details.
    subset_index
        The selected subset of the index. See :class:`~tpcp.Dataset` for details.

    Attributes
    ----------
    data
        The raw IMU data as a dictionary with the MobGap sensor name as key.
    data_ss
        The raw IMU data of the selected recording in sensor frame.
    sampling_rate_hz
        The sampling rate of the selected CWA recording.
    cwa_header_
        CWA metadata with raw reader clock fields and interpreted timestamps in the output timezone.
    cwa_timing_report_
        CWA timing report with raw reader clock fields and interpreted timestamps in the output timezone.
    reference_nonwear_
        Reference non-wear intervals with columns ``start``, ``end``, ``duration``, ``start_dt``, ``end_dt`` and
        ``duration_s``.
    reference_weartime_
        The complement of ``reference_nonwear_`` in the same format.
    """

    base_path: PathLike
    tz: str
    output_timezone: Literal["utc", "local"]
    additional_sensors_enabled: Sequence[AdditionalChannel]
    missing_reference_error_type: MissingReferenceErrorType
    warn_thres_for_sampling_rate_deviations_hz: float | None
    sensor_name: str
    splitter: pd.DataFrame | Callable[[CwaRecordingInfo], pd.DataFrame] | None
    memory: joblib.Memory

    def __init__(
        self,
        base_path: PathLike,
        *,
        tz: str,
        output_timezone: Literal["utc", "local"] = "local",
        additional_sensors_enabled: Sequence[AdditionalChannel] = ("temperature",),
        missing_reference_error_type: MissingReferenceErrorType = "raise",
        warn_thres_for_sampling_rate_deviations_hz: float | None = DEFAULT_WARN_THRES_FOR_SAMPLING_RATE_DEVIATIONS_HZ,
        sensor_name: str = "LowerBack",
        splitter: pd.DataFrame | Callable[[CwaRecordingInfo], pd.DataFrame] | None = cf(_DEFAULT_DAILY_SPLITTER),
        memory: joblib.Memory = joblib.Memory(None),
        groupby_cols: list[str] | str | None = None,
        subset_index: pd.DataFrame | None = None,
    ) -> None:
        self.base_path = base_path
        self.missing_reference_error_type = missing_reference_error_type
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

    @property
    def _human_movement_path(self) -> Path:
        return Path(self.base_path) / "weartime_part_a_all"

    @property
    def _simulated_movements_path(self) -> Path:
        return Path(self.base_path) / "weartime_part_b"

    @property
    def _reference_path(self) -> Path:
        return self._human_movement_path / "reference.json"

    @property
    def recording_metadata(self) -> RecordingMetadata:
        self.assert_is_single(None, "recording_metadata")
        metadata = self.cwa_header_
        row = self.index_as_tuples()[0]
        recording_metadata = {
            "measurement_condition": "laboratory",
            "recording_id": row.recording_id,
            "recording_type": row.recording_type,
            "file_name": self._selected_file_path.name,
            "hardware_type": metadata.get("hardware_type"),
            "device_id": metadata.get("device_id"),
            "logging_start_time": metadata.get("logging_start_time"),
            "logging_end_time": metadata.get("logging_end_time"),
            "cwa_header": metadata,
        }
        recording_metadata["recording_day"] = str(row.recording_day)
        return recording_metadata

    @property
    def participant_metadata(self) -> ParticipantMetadata:
        self.assert_is_single(None, "participant_metadata")
        return {"cohort": None, "height_m": None, "sensor_height_m": None}

    @property
    def reference_nonwear_(self) -> pd.DataFrame:
        self.assert_is_single(None, "reference_nonwear_")
        data = self.data_ss
        if self.index_as_tuples()[0].recording_type == "simulated_movements":
            intervals = pd.DataFrame({"start": [0], "end": [len(data)]})
            return _format_reference_df(
                intervals,
                data_index=data.index,
                sampling_rate_hz=self.sampling_rate_hz,
                index_name="nonwear_id",
            )

        raw_reference = self._raw_reference_for_selected_recording()
        intervals = pd.DataFrame(
            {
                "start": [
                    _timestamp_to_sample_boundary(timestamp, data.index, fallback=0)
                    for timestamp in raw_reference["device_off"]
                ],
                "end": [
                    _timestamp_to_sample_boundary(timestamp, data.index, fallback=len(data))
                    for timestamp in raw_reference["device_on"]
                ],
            }
        )
        return _format_reference_df(
            intervals,
            data_index=data.index,
            sampling_rate_hz=self.sampling_rate_hz,
            index_name="nonwear_id",
        )

    @property
    def reference_weartime_(self) -> pd.DataFrame:
        self.assert_is_single(None, "reference_weartime_")
        data = self.data_ss
        intervals = _complement_intervals(self.reference_nonwear_[["start", "end"]], len(data))
        return _format_reference_df(
            intervals,
            data_index=data.index,
            sampling_rate_hz=self.sampling_rate_hz,
            index_name="weartime_id",
        )

    def _cached_load_reference_file(self) -> pd.DataFrame:
        return hybrid_cache(self.memory, 1)(_load_reference_file)(self._reference_path)

    def _raw_reference_for_selected_recording(self) -> pd.DataFrame:
        participant_id = self.index_as_tuples()[0].participant_id
        reference = self._cached_load_reference_file()
        reference = reference[
            (reference["participant_id"] == participant_id) & (reference["wear_status"] == "non_wear")
        ]

        if reference.empty:
            msg = f"Could not find non-wear reference rows for participant={participant_id}."
            if self.missing_reference_error_type == "raise":
                raise ValueError(msg)
            if self.missing_reference_error_type == "warn":
                warnings.warn(msg, stacklevel=2)

        invalid_timestamp_rows = reference["device_off"].isna()
        if invalid_timestamp_rows.any():
            raise ValueError(
                "The SUSTAIN wear-time reference file contains missing `device_off` timestamps for "
                f"participant={participant_id}. Missing `device_on` timestamps are supported and "
                "treated as open non-wear intervals until the end of the selected recording data."
            )

        return reference

    @property
    def _file_path_root(self) -> Path:
        return Path(self.base_path)

    def _get_file_paths(self) -> list[Path]:
        paths = [
            path
            for folder in (self._human_movement_path, self._simulated_movements_path)
            for path in sorted(folder.glob("*/*.cwa"))
            if _is_lowerback_name(path.name)
        ]
        if not paths:
            raise FileNotFoundError(
                "Could not find any SUSTAIN wear-time lower-back CWA files below "
                f"{self._human_movement_path} or {self._simulated_movements_path}."
            )
        return paths

    def _get_splits_for_file(self, path: Path) -> pd.DataFrame:
        recording_type = "human_movement" if path.parent.parent == self._human_movement_path else "simulated_movements"
        participant_id = path.parent.name
        recording_id = f"{recording_type}_{participant_id}_{path.stem}"
        info = ax6_module._cwa_recording_info(
            path,
            tz=self.tz,
            output_timezone=self.output_timezone,
            recording_metadata={
                "measurement_condition": "laboratory",
                "recording_id": recording_id,
                "recording_type": recording_type,
                "participant_id": participant_id,
                "file_name": path.name,
            },
        )
        if self.splitter is None:
            splits = pd.DataFrame({"recording": ["main"], "start_time": [info.start_time], "end_time": [info.end_time]})
        elif isinstance(self.splitter, pd.DataFrame):
            splits = self.splitter.copy()
        else:
            splits = self.splitter(info)
        return splits.assign(
            recording_type=recording_type,
            participant_id=participant_id,
            recording_id=recording_id,
            recording_day=lambda df_: df_["start_time"].map(
                lambda value: pd.Timestamp(value).tz_convert(self.tz).strftime("%Y-%m-%d")
            ),
        ).astype(
            {
                "recording_type": "string",
                "participant_id": "string",
                "recording_id": "string",
                "recording_day": "string",
            }
        )


__all__ = ["SustainWearTimeDataset"]
