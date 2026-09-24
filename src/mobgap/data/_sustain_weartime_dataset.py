"""Dataset loader for the SUSTAIN wear-time recordings."""

from __future__ import annotations

import re
import warnings
from math import floor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Union

import joblib
import pandas as pd
from tpcp.caching import hybrid_cache

from mobgap.consts import GRAV_MS2, SF_ACC_COLS, SF_SENSOR_COLS
from mobgap.data.base import IMU_DATA_DTYPE, BaseGaitDataset, ParticipantMetadata, RecordingMetadata

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    from cwa_reader_rs import read_cwa_file, read_header, sampling_consistency_report, seconds
except ImportError:
    read_cwa_file = None
    read_header = None
    sampling_consistency_report = None
    seconds = None

PathLike = Union[str, Path]
MissingReferenceErrorType = Literal["raise", "warn", "ignore"]
AdditionalCwaChannel = Literal["temperature", "light", "battery"]

REFERENCE_COLUMNS = ["start", "end", "duration", "start_dt", "end_dt", "duration_s"]
ADDITIONAL_CWA_CHANNELS: tuple[AdditionalCwaChannel, ...] = ("temperature", "light", "battery")
ADDITIONAL_CWA_OUTPUT_COLUMNS: dict[AdditionalCwaChannel, tuple[str, ...]] = {
    "temperature": ("temperature",),
    "light": ("light",),
    "battery": ("battery",),
}
DEFAULT_WARN_THRES_FOR_SAMPLING_RATE_DEVIATIONS_HZ = 0.2
SAMPLING_RATE_DEVIATION_WARNING = (
    "The expected number of samples/the effective sampling rate waries considerable from the expected values. "
    "While this is likely normal and might happen due to clock drift in long recordings, it might be worth "
    "investigating further. Data will be resampled assuming that the recorded start and end dates are correct."
)
CWA_READER_IMPORT_ERROR = (
    "The optional dependency `cwa_reader_rs` is required to load SUSTAIN wear-time CWA files. "
    "Install MobGap with the optional wear-time dependencies before using `SustainWearTimeDataset`."
)


class _CwaRecording(NamedTuple):
    data: pd.DataFrame
    sampling_rate_hz: float
    metadata: dict[str, Any]
    timing_report: dict[str, Any]


def _is_lowerback_name(name: str) -> bool:
    name = name.lower()
    compact_name = name.replace("_", "").replace("-", "").replace(" ", "")
    tokens = {token for token in re.split(r"[\W_]+", name) if token}
    return "lowerback" in compact_name or "lowback" in compact_name or "lb" in tokens


def _normalize_additional_channels(
    additional_channels: Sequence[AdditionalCwaChannel],
) -> tuple[AdditionalCwaChannel, ...]:
    unique_channels = tuple(dict.fromkeys(additional_channels))
    unknown_channels = set(unique_channels) - set(ADDITIONAL_CWA_CHANNELS)
    if unknown_channels:
        raise ValueError(
            "Unknown additional CWA channels. "
            f"Unknown: {sorted(unknown_channels)}. Available: {list(ADDITIONAL_CWA_CHANNELS)}."
        )
    return unique_channels


def _available_additional_channels() -> tuple[AdditionalCwaChannel, ...]:
    return ADDITIONAL_CWA_CHANNELS


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


def _read_cwa_header(file_path: PathLike) -> dict[str, Any]:
    if read_header is None:
        raise ImportError(CWA_READER_IMPORT_ERROR)

    return dict(read_header(str(file_path)))


def _read_cwa_timing_report(file_path: PathLike) -> dict[str, Any]:
    if sampling_consistency_report is None:
        raise ImportError(CWA_READER_IMPORT_ERROR)

    return dict(sampling_consistency_report(str(file_path)))


def _warn_if_effective_sampling_rate_deviates(
    timing_report: dict[str, Any],
    threshold_hz: float | None,
) -> None:
    if threshold_hz is None:
        return
    expected_sampling_rate_hz = timing_report.get("samplingrate_hz_from_header")
    effective_sampling_rate_hz = timing_report.get("samplingrate_hz_from_data")
    if expected_sampling_rate_hz is None or effective_sampling_rate_hz is None:
        return
    if abs(float(effective_sampling_rate_hz) - float(expected_sampling_rate_hz)) > threshold_hz:
        warnings.warn(SAMPLING_RATE_DEVIATION_WARNING, stacklevel=2)


def _read_cwa_recording(
    file_path: PathLike,
    additional_channels: Sequence[AdditionalCwaChannel],
    timing_report: dict[str, Any],
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> _CwaRecording:
    if read_cwa_file is None:
        raise ImportError(CWA_READER_IMPORT_ERROR)

    additional_channels = _normalize_additional_channels(additional_channels)
    metadata = _read_cwa_header(file_path)
    sampling_rate_hz = float(timing_report.get("samplingrate_hz_from_header") or metadata["sample_rate_hz"])
    cut = None if start_time_s is None and end_time_s is None else seconds(start_time_s, end_time_s)
    raw_data = pd.DataFrame(
        read_cwa_file(
            str(file_path),
            cut=cut,
            include_magnetometer=False,
            include_temperature="temperature" in additional_channels,
            include_light="light" in additional_channels,
            include_battery="battery" in additional_channels,
            resample_hz=sampling_rate_hz,
            resample_method="cubic",
        )
    )

    if "timestamp" not in raw_data.columns:
        raise ValueError(f"The CWA reader did not return a `timestamp` column for {file_path}.")

    index = pd.DatetimeIndex(pd.to_datetime(raw_data.pop("timestamp").to_numpy(), unit="us", utc=True), name="time")
    data = raw_data.rename(columns={"gyro_x": "gyr_x", "gyro_y": "gyr_y", "gyro_z": "gyr_z"})
    data.index = index

    missing_sensor_columns = set(SF_SENSOR_COLS) - set(data.columns)
    if missing_sensor_columns:
        raise ValueError(
            "The CWA reader did not return all expected sensor-frame columns. "
            f"Missing columns: {sorted(missing_sensor_columns)}."
        )

    additional_output_columns = [
        column
        for channel in ADDITIONAL_CWA_CHANNELS
        if channel in additional_channels
        for column in ADDITIONAL_CWA_OUTPUT_COLUMNS[channel]
    ]
    missing_additional_columns = set(additional_output_columns) - set(data.columns)
    if missing_additional_columns:
        raise ValueError(
            "The CWA reader did not return all requested additional channel columns. "
            f"Missing columns: {sorted(missing_additional_columns)}."
        )

    output_columns = [*SF_SENSOR_COLS, *additional_output_columns]
    data = data[output_columns].copy()
    data[SF_ACC_COLS] *= GRAV_MS2

    return _CwaRecording(
        data=data,
        sampling_rate_hz=sampling_rate_hz,
        metadata=metadata,
        timing_report=timing_report,
    )


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
    return max(0, floor((end_time_s - start_time_s) * sampling_rate_hz + 1e-9))


def _clip_data_to_recording_day(data: pd.DataFrame, recording_day: str | None) -> pd.DataFrame:
    if recording_day is None or not isinstance(data.index, pd.DatetimeIndex):
        return data

    day_start = _as_utc_timestamp(recording_day)
    day_start = day_start.tz_localize(None) if data.index.tz is None else day_start.tz_convert(data.index.tz)
    day_end = day_start + pd.Timedelta(days=1)

    return data.loc[(data.index >= day_start) & (data.index < day_end)]


class SustainWearTimeDataset(BaseGaitDataset):
    """Dataset for the SUSTAIN wear-time raw CWA recordings.

    The dataset index contains one row per raw CWA recording. The raw data is loaded lazily and returned in the MobGap
    sensor frame. Reference intervals use MobGap's half-open sample convention: ``start`` is inclusive and ``end`` is
    exclusive. ``start_dt`` and ``end_dt`` are derived from the snapped sample boundaries, not copied from the raw
    reference file.

    Parameters
    ----------
    base_path
        The root folder containing ``weartime_part_a_all`` and ``weartime_part_b``.
    additional_channels
        Additional CWA channels to append to the core accelerometer and gyroscope data. Potential channels are
        ``"temperature"``, ``"light"`` and ``"battery"``. The dataset validates that the selected recording actually
        contains each requested channel.
    missing_reference_error_type
        How to handle missing part A reference rows for a selected recording.
    warn_thres_for_sampling_rate_deviations_hz
        Threshold in Hz used to warn when the effective sampling rate differs from the expected sampling rate. Set to
        ``None`` to disable the warning.
    split_by_day
        If ``True``, the dataset index contains one row per calendar day spanned by a raw CWA recording. The
        ``recording_day`` column identifies the selected day and ``data_ss`` loads only the respective time window.
        If ``False``, the index contains one row per raw CWA recording.
    memory
        A joblib memory object used to cache CWA header, timing report, CWA data, and reference file loading.
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
    available_additional_channels_
        Additional CWA channels available for the selected recording.
    n_samples
        Number of samples in the selected recording, derived from CWA timing metadata without loading the full data.
    reference_nonwear_
        Reference non-wear intervals with columns ``start``, ``end``, ``duration``, ``start_dt``, ``end_dt`` and
        ``duration_s``.
    reference_weartime_
        The complement of ``reference_nonwear_`` in the same format.
    """

    base_path: PathLike
    additional_channels: Sequence[AdditionalCwaChannel]
    missing_reference_error_type: MissingReferenceErrorType
    warn_thres_for_sampling_rate_deviations_hz: float | None
    split_by_day: bool
    memory: joblib.Memory

    def __init__(
        self,
        base_path: PathLike,
        *,
        additional_channels: Sequence[AdditionalCwaChannel] = ("temperature",),
        missing_reference_error_type: MissingReferenceErrorType = "raise",
        warn_thres_for_sampling_rate_deviations_hz: float | None = DEFAULT_WARN_THRES_FOR_SAMPLING_RATE_DEVIATIONS_HZ,
        split_by_day: bool = False,
        memory: joblib.Memory = joblib.Memory(None),
        groupby_cols: list[str] | str | None = None,
        subset_index: pd.DataFrame | None = None,
    ) -> None:
        self.base_path = base_path
        self.additional_channels = additional_channels
        self.missing_reference_error_type = missing_reference_error_type
        self.warn_thres_for_sampling_rate_deviations_hz = warn_thres_for_sampling_rate_deviations_hz
        self.split_by_day = split_by_day
        self.memory = memory
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

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
    def _additional_channels(self) -> tuple[AdditionalCwaChannel, ...]:
        return _normalize_additional_channels(self.additional_channels)

    @property
    def selected_data_file(self) -> Path:
        self.assert_is_single(None, "selected_data_file")
        row = self.index.iloc[0]
        file_index = self._recording_file_index()
        matches = file_index[
            (file_index["recording_type"] == row["recording_type"])
            & (file_index["participant_id"] == row["participant_id"])
            & (file_index["recording_id"] == row["recording_id"])
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Could not uniquely resolve the selected CWA file for {self.group_label}.")
        return Path(matches.iloc[0]["file_path"])

    @property
    def _selected_recording_day(self) -> str | None:
        self.assert_is_single(None, "_selected_recording_day")
        if "recording_day" not in self.index.columns:
            return None
        return str(self.index.iloc[0]["recording_day"])

    @property
    def data(self) -> IMU_DATA_DTYPE:
        self.assert_is_single(None, "data")
        return {"LowerBack": self.data_ss}

    @property
    def data_ss(self) -> pd.DataFrame:
        self.assert_is_single(None, "data_ss")
        additional_channels = self._additional_channels
        timing_report = self.cwa_timing_report_
        _warn_if_effective_sampling_rate_deviates(timing_report, self.warn_thres_for_sampling_rate_deviations_hz)
        start_time_s, end_time_s = _day_cut_seconds(self._selected_recording_day, timing_report, self.sampling_rate_hz)
        data = self._cached_load_cwa_recording(
            self.selected_data_file, additional_channels, timing_report, start_time_s, end_time_s
        ).data
        return _clip_data_to_recording_day(data, self._selected_recording_day)

    @property
    def sampling_rate_hz(self) -> float:
        self.assert_is_single(None, "sampling_rate_hz")
        return float(self.cwa_header_["sample_rate_hz"])

    @property
    def cwa_header_(self) -> dict[str, Any]:
        self.assert_is_single(None, "cwa_header_")
        return dict(self._cached_load_cwa_header(self.selected_data_file))

    @property
    def cwa_timing_report_(self) -> dict[str, Any]:
        self.assert_is_single(None, "cwa_timing_report_")
        return dict(self._cached_load_cwa_timing_report(self.selected_data_file))

    @property
    def available_additional_channels_(self) -> tuple[AdditionalCwaChannel, ...]:
        self.assert_is_single(None, "available_additional_channels_")
        return _available_additional_channels()

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
            "file_name": self.selected_data_file.name,
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

    def _cached_load_cwa_header(self, file_path: PathLike) -> dict[str, Any]:
        return hybrid_cache(self.memory, 1)(_read_cwa_header)(file_path)

    def _cached_load_cwa_timing_report(self, file_path: PathLike) -> dict[str, Any]:
        return hybrid_cache(self.memory, 1)(_read_cwa_timing_report)(file_path)

    def _cached_load_cwa_recording(
        self,
        file_path: PathLike,
        additional_channels: Sequence[AdditionalCwaChannel],
        timing_report: dict[str, Any],
        start_time_s: float | None = None,
        end_time_s: float | None = None,
    ) -> _CwaRecording:
        return hybrid_cache(self.memory, 1)(_read_cwa_recording)(
            file_path, additional_channels, timing_report, start_time_s, end_time_s
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

    def _recording_file_index(self) -> pd.DataFrame:
        rows = []

        for recording_type, recording_path in (
            ("human_movement", self._human_movement_path),
            ("simulated_movements", self._simulated_movements_path),
        ):
            if not recording_path.exists():
                continue
            for file_path in sorted(recording_path.glob("*/*.cwa")):
                if not _is_lowerback_name(file_path.name):
                    continue
                participant_id = file_path.parent.name
                rows.append(
                    {
                        "recording_type": recording_type,
                        "participant_id": participant_id,
                        "recording_id": f"{recording_type}_{participant_id}_{file_path.stem}",
                        "file_path": file_path,
                    }
                )

        if not rows:
            raise FileNotFoundError(
                "Could not find any SUSTAIN wear-time lower-back CWA files below "
                f"{self._human_movement_path} or {self._simulated_movements_path}."
            )

        return pd.DataFrame(rows).sort_values(["recording_type", "participant_id", "recording_id"], ignore_index=True)

    def create_index(self) -> pd.DataFrame:
        file_index = self._recording_file_index()
        if not self.split_by_day:
            return file_index.drop(columns=["file_path"]).astype("string")

        rows: list[dict[str, Any]] = []
        for row in file_index.to_dict("records"):
            timing_report = self._cached_load_cwa_timing_report(row["file_path"])
            for recording_day in _recording_days_from_timing_report(timing_report):
                rows.append({**row, "recording_day": recording_day})

        return pd.DataFrame(rows).drop(columns=["file_path"]).astype("string")


__all__ = ["SustainWearTimeDataset"]
