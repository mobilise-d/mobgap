"""Dataset loader for the SUSTAIN wear-time recordings."""

from __future__ import annotations

import re
import warnings
from math import ceil, floor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union

import joblib
import pandas as pd
from tpcp.caching import hybrid_cache

from mobgap.data import ax6 as ax6_module
from mobgap.data.ax6 import AdditionalChannel, BaseAX6Dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mobgap.data.base import ParticipantMetadata, RecordingMetadata

PathLike = Union[str, Path]
MissingReferenceErrorType = Literal["raise", "warn", "ignore"]
REFERENCE_COLUMNS = ["start", "end", "duration", "start_dt", "end_dt", "duration_s"]
DEFAULT_WARN_THRES_FOR_SAMPLING_RATE_DEVIATIONS_HZ = 0.2
SAMPLE_COUNT_TOL = 1e-6


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
            device_off=lambda df_: pd.to_datetime(df_["device_off"], errors="raise", utc=True),
            device_on=lambda df_: pd.to_datetime(df_["device_on"], errors="raise", utc=True),
        )
        .drop(columns=["id", "sensor", "is_lowerback"])
    )

    return reference.sort_values(["participant_id", "device_off", "device_on"], ignore_index=True)


def _empty_reference_df(index_name: str, datetime_dtype: str = "datetime64[ns, UTC]") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start": pd.Series(dtype="int64"),
            "end": pd.Series(dtype="int64"),
            "duration": pd.Series(dtype="int64"),
            "start_dt": pd.Series(dtype=datetime_dtype),
            "end_dt": pd.Series(dtype=datetime_dtype),
            "duration_s": pd.Series(dtype="float64"),
        }
    ).rename_axis(index_name)


def _sample_boundary_timestamp(
    data_index: pd.DatetimeIndex, sample_boundary: int, sampling_rate_hz: float
) -> pd.Timestamp:
    if len(data_index) == 0:
        return pd.NaT
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


def _merge_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    if intervals.empty:
        return intervals.astype({"start": "int64", "end": "int64"})

    intervals = intervals.sort_values(["start", "end"], ignore_index=True).astype({"start": "int64", "end": "int64"})
    merged: list[tuple[int, int]] = []
    for start, end in intervals[["start", "end"]].itertuples(index=False):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((int(start), int(end)))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], int(end)))
    return pd.DataFrame(merged, columns=["start", "end"])


def _format_reference_df(
    intervals: pd.DataFrame,
    *,
    data_index: pd.DatetimeIndex,
    sampling_rate_hz: float,
    index_name: str,
) -> pd.DataFrame:
    intervals = _merge_intervals(intervals)
    if intervals.empty:
        return _empty_reference_df(index_name)

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
    for start, end in _merge_intervals(intervals)[["start", "end"]].itertuples(index=False):
        if start > current_start:
            complement.append((current_start, int(start)))
        current_start = max(current_start, int(end))
    if current_start < data_length:
        complement.append((current_start, data_length))
    return pd.DataFrame(complement, columns=["start", "end"])


def _recording_start_end_from_timing_report(timing_report: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = timing_report.get("start_from_data")
    end = timing_report.get("end_from_data")
    if start is None or end is None:
        raise ValueError(
            "The CWA timing report does not contain `start_from_data` and `end_from_data`. "
            "These fields are required to split SUSTAIN wear-time recordings by day."
        )

    start = _as_utc_timestamp(start)
    end = _as_utc_timestamp(end)
    if end < start:
        raise ValueError(
            f"The CWA timing report contains an `end_from_data` timestamp before `start_from_data`: {end} < {start}."
        )
    return start, end


def _recording_days_from_timing_report(timing_report: dict[str, Any]) -> list[str]:
    start, end = _recording_start_end_from_timing_report(timing_report)
    days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    return [day.date().isoformat() for day in days]


def _day_cut_seconds(
    recording_day: str | None, timing_report: dict[str, Any], sampling_rate_hz: float
) -> tuple[float | None, float | None]:
    if recording_day is None:
        return None, None

    recording_start, recording_end = _recording_start_end_from_timing_report(timing_report)
    recording_end_exclusive = recording_end + pd.to_timedelta(1 / sampling_rate_hz, unit="s")
    day_start = _as_utc_timestamp(recording_day)
    day_end = day_start + pd.Timedelta(days=1)

    start = max(day_start, recording_start)
    end = min(day_end, recording_end_exclusive)
    if end <= start:
        raise ValueError(f"The selected day {recording_day} does not overlap the selected CWA recording.")

    start_time_s = None if start <= recording_start else (start - recording_start).total_seconds()
    end_time_s = None if end >= recording_end_exclusive else (end - recording_start).total_seconds()
    return start_time_s, end_time_s


def _recording_sample_count_from_timing_report(
    timing_report: dict[str, Any], sampling_rate_hz: float, recording_day: str | None
) -> int:
    duration_s = timing_report.get("duration_s_from_data")
    if duration_s is None:
        raise ValueError("The CWA timing report does not contain `duration_s_from_data`.")

    recording_duration_exclusive_s = float(duration_s) + 1 / sampling_rate_hz
    if recording_day is None:
        return max(0, floor(recording_duration_exclusive_s * sampling_rate_hz + 1e-9))

    start_time_s, end_time_s = _day_cut_seconds(recording_day, timing_report, sampling_rate_hz)
    start_time_s = 0.0 if start_time_s is None else start_time_s
    end_time_s = recording_duration_exclusive_s if end_time_s is None else end_time_s
    return _sample_count_from_time_bounds_s(start_time_s, end_time_s, sampling_rate_hz)


def _sample_count_from_time_bounds_s(start_time_s: float, end_time_s: float, sampling_rate_hz: float) -> int:
    start_sample = ceil(start_time_s * sampling_rate_hz - SAMPLE_COUNT_TOL)
    end_sample = ceil(end_time_s * sampling_rate_hz - SAMPLE_COUNT_TOL)
    return max(0, end_sample - start_sample)


class SustainWearTimeDataset(BaseAX6Dataset):
    """Dataset for the SUSTAIN wear-time raw CWA recordings.

    The dataset index contains one row per recording or recording day, including its ``file_path``. The raw data is
    loaded lazily and returned in the MobGap sensor frame. Reference intervals use MobGap's half-open sample convention:
    ``start`` is inclusive and ``end`` is exclusive. ``start_dt`` and ``end_dt`` are derived from the snapped
    sample boundaries, not copied from the raw reference file.

    Parameters
    ----------
    base_path
        The root folder containing ``weartime_part_a_all`` and ``weartime_part_b``.
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
    split_by_day
        If ``True``, the dataset index contains one row per calendar day spanned by a raw CWA recording. The
        ``recording_day`` column identifies the selected day and ``data_ss`` loads only the respective time window.
        If ``False``, the index contains one row per raw CWA recording.
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
        The full CWA header of the selected recording as returned by ``cwa_reader_rs``.
    cwa_timing_report_
        The CWA timing report of the selected recording as returned by ``cwa_reader_rs``.
    n_samples
        Number of samples in the selected recording, derived from CWA timing metadata without loading the full data.
    reference_nonwear_
        Reference non-wear intervals with columns ``start``, ``end``, ``duration``, ``start_dt``, ``end_dt`` and
        ``duration_s``.
    reference_weartime_
        The complement of ``reference_nonwear_`` in the same format.
    """

    base_path: PathLike
    additional_sensors_enabled: Sequence[AdditionalChannel]
    missing_reference_error_type: MissingReferenceErrorType
    warn_thres_for_sampling_rate_deviations_hz: float | None
    sensor_name: str
    split_by_day: bool
    memory: joblib.Memory

    def __init__(
        self,
        base_path: PathLike,
        *,
        additional_sensors_enabled: Sequence[AdditionalChannel] = ("temperature",),
        missing_reference_error_type: MissingReferenceErrorType = "raise",
        warn_thres_for_sampling_rate_deviations_hz: float | None = DEFAULT_WARN_THRES_FOR_SAMPLING_RATE_DEVIATIONS_HZ,
        sensor_name: str = "LowerBack",
        split_by_day: bool = False,
        memory: joblib.Memory = joblib.Memory(None),
        groupby_cols: list[str] | str | None = None,
        subset_index: pd.DataFrame | None = None,
    ) -> None:
        self.base_path = base_path
        self.missing_reference_error_type = missing_reference_error_type
        self.split_by_day = split_by_day
        super().__init__(
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
    def _selected_recording_day(self) -> str | None:
        self.assert_is_single(None, "_selected_recording_day")
        if "recording_day" not in self.index.columns:
            return None
        return str(self.index.iloc[0]["recording_day"])

    def _selected_time_bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        start, last_sample = _recording_start_end_from_timing_report(self.cwa_timing_report_)
        end = last_sample + pd.to_timedelta(1 / self.sampling_rate_hz, unit="s")
        if (recording_day := self._selected_recording_day) is None:
            return start, end
        day_start = _as_utc_timestamp(recording_day)
        return max(start, day_start), min(end, day_start + pd.Timedelta(days=1))

    @property
    def n_samples(self) -> int:
        self.assert_is_single(None, "n_samples")
        return _recording_sample_count_from_timing_report(
            self.cwa_timing_report_,
            self.sampling_rate_hz,
            self._selected_recording_day,
        )

    @property
    def recording_metadata(self) -> RecordingMetadata:
        self.assert_is_single(None, "recording_metadata")
        metadata = self.cwa_header_
        recording_metadata = {
            "measurement_condition": "laboratory",
            "recording_id": self.group_label.recording_id,
            "recording_type": self.group_label.recording_type,
            "file_name": self._selected_file_path.name,
            "hardware_type": metadata.get("hardware_type"),
            "device_id": metadata.get("device_id"),
            "logging_start_time": metadata.get("logging_start_time"),
            "logging_end_time": metadata.get("logging_end_time"),
            "cwa_header": metadata,
        }
        if (recording_day := self._selected_recording_day) is not None:
            recording_metadata["recording_day"] = recording_day
        return recording_metadata

    @property
    def participant_metadata(self) -> ParticipantMetadata:
        self.assert_is_single(None, "participant_metadata")
        return {"cohort": None, "height_m": None, "sensor_height_m": None}

    @property
    def reference_nonwear_(self) -> pd.DataFrame:
        self.assert_is_single(None, "reference_nonwear_")
        data = self.data_ss
        if self.group_label.recording_type == "simulated_movements":
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
        intervals["start"] = intervals["start"].clip(0, len(data))
        intervals["end"] = intervals["end"].clip(0, len(data))
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
        reference = self._cached_load_reference_file()
        reference = reference[
            (reference["participant_id"] == self.group_label.participant_id) & (reference["wear_status"] == "non_wear")
        ]

        if reference.empty:
            msg = f"Could not find non-wear reference rows for participant={self.group_label.participant_id}."
            if self.missing_reference_error_type == "raise":
                raise ValueError(msg)
            if self.missing_reference_error_type == "warn":
                warnings.warn(msg, stacklevel=2)

        invalid_timestamp_rows = reference["device_off"].isna()
        if invalid_timestamp_rows.any():
            raise ValueError(
                "The SUSTAIN wear-time reference file contains missing `device_off` timestamps for "
                f"participant={self.group_label.participant_id}. Missing `device_on` timestamps are supported and "
                "treated as open non-wear intervals until the end of the selected recording data."
            )

        return reference.sort_values(["device_off", "device_on"], ignore_index=True)

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
        recording = {
            "recording_type": recording_type,
            "participant_id": participant_id,
            "recording_id": f"{recording_type}_{participant_id}_{path.stem}",
        }
        if not self.split_by_day:
            return pd.DataFrame([recording]).astype("string")

        timing_report = ax6_module._recording_info(path, ax6_module._file_identity(path))[1]
        return pd.DataFrame(
            [{**recording, "recording_day": day} for day in _recording_days_from_timing_report(timing_report)]
        ).astype("string")


__all__ = ["SustainWearTimeDataset"]
