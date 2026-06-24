"""Dataset loader for the SUSTAIN wear-time recordings."""

from __future__ import annotations

import warnings
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
    from cwa_reader_rs import read_cwa_file, read_header
except ImportError:
    read_cwa_file = None
    read_header = None

PathLike = Union[str, Path]
MissingReferenceErrorType = Literal["raise", "warn", "ignore"]

REFERENCE_COLUMNS = ["start", "end", "duration", "start_dt", "end_dt", "duration_s"]


class _CwaRecording(NamedTuple):
    data: pd.DataFrame
    sampling_rate_hz: float
    metadata: dict[str, Any]


def _normalize_sensor_position(sensor_position: str) -> str:
    normalized = sensor_position.lower().replace("_", "").replace("-", "").replace(" ", "")
    if normalized in {"lowerback", "lowback", "lb"}:
        return "lowerback"
    if normalized in {"wrist", "wr"}:
        return "wrist"
    return normalized


def _sensor_position_from_file_name(file_path: Path) -> str:
    file_name = file_path.name.lower()
    for sensor_position in ("lowerback", "lowback", "lb", "wrist", "wr"):
        if sensor_position in file_name:
            return _normalize_sensor_position(sensor_position)
    raise ValueError(f"Could not infer the sensor position from the file name: {file_path.name}.")


def _data_key_from_sensor_position(sensor_position: str) -> str:
    mapping = {"lowerback": "LowerBack", "wrist": "Wrist"}
    return mapping.get(sensor_position, sensor_position)


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
        return pd.DataFrame(columns=["participant_id", "sensor_position", "device_off", "device_on", "wear_status"])

    required_columns = {"id", "sensor", "device_off", "device_on", "wear_status"}
    missing_columns = required_columns - set(reference.columns)
    if missing_columns:
        raise ValueError(
            f"The SUSTAIN wear-time reference file is missing the following columns: {sorted(missing_columns)}."
        )

    return (
        reference.assign(
            participant_id=lambda df_: df_["id"].astype("string").str.zfill(3),
            sensor_position=lambda df_: df_["sensor"].map(_normalize_sensor_position).astype("string"),
            device_off=lambda df_: pd.to_datetime(df_["device_off"], errors="coerce", utc=True),
            device_on=lambda df_: pd.to_datetime(df_["device_on"], errors="coerce", utc=True),
            wear_status=lambda df_: df_["wear_status"].astype("string"),
        )
        .drop(columns=["id", "sensor"])
        .sort_values(["participant_id", "sensor_position", "device_off", "device_on"], ignore_index=True)
    )


def _read_cwa_header(file_path: PathLike) -> dict[str, Any]:
    if read_header is None:
        raise ImportError(
            "The optional dependency `cwa_reader_rs` is required to load SUSTAIN wear-time CWA files. "
            "Install it with `uv sync --extra weartime`."
        )

    return dict(read_header(str(file_path)))


def _read_cwa_recording(file_path: PathLike) -> _CwaRecording:
    if read_cwa_file is None:
        raise ImportError(
            "The optional dependency `cwa_reader_rs` is required to load SUSTAIN wear-time CWA files. "
            "Install it with `uv sync --extra weartime`."
        )

    metadata = _read_cwa_header(file_path)
    raw_data = pd.DataFrame(
        read_cwa_file(
            str(file_path),
            include_magnetometer=False,
            include_temperature=True,
            include_light=False,
            include_battery=False,
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

    output_columns = [*SF_SENSOR_COLS, *(["temperature"] if "temperature" in data.columns else [])]
    data = data[output_columns].copy()
    data[SF_ACC_COLS] *= GRAV_MS2

    return _CwaRecording(
        data=data,
        sampling_rate_hz=float(metadata["sample_rate_hz"]),
        metadata=metadata,
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
    human_movement_path
        Optional override for the folder containing the part A recordings.
    simulated_movements_path
        Optional override for the folder containing the part B recordings.
    reference_path
        Optional override for the part A JSON-lines reference file.
    sensor_positions
        Optional list of sensor positions to include. Values are normalized to the reference-file naming convention,
        e.g. ``"lowerback"``.
    missing_reference_error_type
        How to handle missing part A reference rows for a selected recording.
    memory
        A joblib memory object used to cache CWA header, CWA data, and reference file loading.
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
    reference_nonwear_
        Reference non-wear intervals with columns ``start``, ``end``, ``duration``, ``start_dt``, ``end_dt`` and
        ``duration_s``.
    reference_weartime_
        The complement of ``reference_nonwear_`` in the same format.
    """

    base_path: PathLike
    human_movement_path: PathLike | None
    simulated_movements_path: PathLike | None
    reference_path: PathLike | None
    sensor_positions: Sequence[str] | None
    missing_reference_error_type: MissingReferenceErrorType
    memory: joblib.Memory

    def __init__(
        self,
        base_path: PathLike,
        *,
        human_movement_path: PathLike | None = None,
        simulated_movements_path: PathLike | None = None,
        reference_path: PathLike | None = None,
        sensor_positions: Sequence[str] | None = None,
        missing_reference_error_type: MissingReferenceErrorType = "raise",
        memory: joblib.Memory = joblib.Memory(None),
        groupby_cols: list[str] | str | None = None,
        subset_index: pd.DataFrame | None = None,
    ) -> None:
        self.base_path = base_path
        self.human_movement_path = human_movement_path
        self.simulated_movements_path = simulated_movements_path
        self.reference_path = reference_path
        self.sensor_positions = sensor_positions
        self.missing_reference_error_type = missing_reference_error_type
        self.memory = memory
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def _human_movement_path(self) -> Path:
        return (
            Path(self.human_movement_path)
            if self.human_movement_path is not None
            else Path(self.base_path) / ("weartime_part_a_all")
        )

    @property
    def _simulated_movements_path(self) -> Path:
        return (
            Path(self.simulated_movements_path)
            if self.simulated_movements_path is not None
            else Path(self.base_path) / "weartime_part_b"
        )

    @property
    def _reference_path(self) -> Path:
        return (
            Path(self.reference_path)
            if self.reference_path is not None
            else self._human_movement_path / "reference.json"
        )

    @property
    def selected_data_file(self) -> Path:
        self.assert_is_single(None, "selected_data_file")
        row = self.index.iloc[0]
        file_index = self._recording_file_index()
        matches = file_index[
            (file_index["recording_type"] == row["recording_type"])
            & (file_index["participant_id"] == row["participant_id"])
            & (file_index["sensor_position"] == row["sensor_position"])
            & (file_index["recording_id"] == row["recording_id"])
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Could not uniquely resolve the selected CWA file for {self.group_label}.")
        return Path(matches.iloc[0]["file_path"])

    @property
    def data(self) -> IMU_DATA_DTYPE:
        self.assert_is_single(None, "data")
        return {_data_key_from_sensor_position(self.group_label.sensor_position): self.data_ss}

    @property
    def data_ss(self) -> pd.DataFrame:
        self.assert_is_single(None, "data_ss")
        return self._cached_load_cwa_recording(self.selected_data_file).data

    @property
    def sampling_rate_hz(self) -> float:
        self.assert_is_single(None, "sampling_rate_hz")
        return float(self._cached_load_cwa_header(self.selected_data_file)["sample_rate_hz"])

    @property
    def recording_metadata(self) -> RecordingMetadata:
        self.assert_is_single(None, "recording_metadata")
        metadata = self._cached_load_cwa_header(self.selected_data_file)
        return {
            "measurement_condition": "laboratory",
            "recording_id": self.group_label.recording_id,
            "recording_type": self.group_label.recording_type,
            "sensor_position": self.group_label.sensor_position,
            "file_name": self.selected_data_file.name,
            "hardware_type": metadata.get("hardware_type"),
            "device_id": metadata.get("device_id"),
            "logging_start_time": metadata.get("logging_start_time"),
            "logging_end_time": metadata.get("logging_end_time"),
        }

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

    def _cached_load_cwa_recording(self, file_path: PathLike) -> _CwaRecording:
        return hybrid_cache(self.memory, 1)(_read_cwa_recording)(file_path)

    def _cached_load_reference_file(self) -> pd.DataFrame:
        return hybrid_cache(self.memory, 1)(_load_reference_file)(self._reference_path)

    def _raw_reference_for_selected_recording(self) -> pd.DataFrame:
        reference = self._cached_load_reference_file()
        reference = reference[
            (reference["participant_id"] == self.group_label.participant_id)
            & (reference["sensor_position"] == self.group_label.sensor_position)
            & (reference["wear_status"] == "non_wear")
        ]

        if reference.empty:
            msg = (
                "Could not find non-wear reference rows for "
                f"participant={self.group_label.participant_id}, "
                f"sensor_position={self.group_label.sensor_position}."
            )
            if self.missing_reference_error_type == "raise":
                raise ValueError(msg)
            if self.missing_reference_error_type == "warn":
                warnings.warn(msg, stacklevel=2)

        return reference.sort_values(["device_off", "device_on"], ignore_index=True)

    def _recording_file_index(self) -> pd.DataFrame:
        rows = []
        sensor_position_filter = (
            None if self.sensor_positions is None else {_normalize_sensor_position(s) for s in self.sensor_positions}
        )

        for recording_type, recording_path in (
            ("human_movement", self._human_movement_path),
            ("simulated_movements", self._simulated_movements_path),
        ):
            if not recording_path.exists():
                continue
            for file_path in sorted(recording_path.glob("*/*.cwa")):
                sensor_position = _sensor_position_from_file_name(file_path)
                if sensor_position_filter is not None and sensor_position not in sensor_position_filter:
                    continue
                participant_id = file_path.parent.name
                rows.append(
                    {
                        "recording_type": recording_type,
                        "participant_id": participant_id,
                        "sensor_position": sensor_position,
                        "recording_id": f"{recording_type}_{participant_id}_{sensor_position}",
                        "file_path": file_path,
                    }
                )

        if not rows:
            raise FileNotFoundError(
                "Could not find any SUSTAIN wear-time CWA files below "
                f"{self._human_movement_path} or {self._simulated_movements_path}."
            )

        return pd.DataFrame(rows).sort_values(
            ["recording_type", "participant_id", "sensor_position", "recording_id"], ignore_index=True
        )

    def create_index(self) -> pd.DataFrame:
        return self._recording_file_index().drop(columns=["file_path"]).astype("string")


__all__ = ["SustainWearTimeDataset"]
