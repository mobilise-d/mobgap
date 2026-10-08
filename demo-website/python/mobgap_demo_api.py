"""Browser-worker bridge to mobgap MATLAB/CWA loaders and full pipelines.

Acceleration conversion and coordinate handling belong to mobgap's loaders
and pipeline. This module never rescales or reorients sensor data.
"""

from __future__ import annotations

import gc
import importlib.metadata
import json
import math
import time
import traceback
import uuid
import warnings
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pandas as pd
from mobgap.data import AX6Dataset, GenericMobilisedDataset, load_mobilised_participant_metadata_file
from mobgap.data.ax6 import split_by_local_days
from mobgap.pipeline import MobilisedPipelineHealthy, MobilisedPipelineImpaired, MobilisedPipelineUniversal
from scipy.io import whosmat

_REQUIRED_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")
_RECORDINGS: dict[str, dict[str, Any]] = {}
_DAY_BATCH: Generator[dict[str, Any]] | None = None


def _mat_variables(path: Path) -> set[str]:
    with path.open("rb") as file:
        header = file.read(128)
    if header.startswith(b"\x89HDF\r\n\x1a\n") or b"MATLAB 7.3 MAT-file" in header:
        raise ValueError("MATLAB v7.3/HDF5 files are not supported. Save with MATLAB's -v7 option and try again.")
    try:
        return {name for name, _shape, _dtype in whosmat(path)}
    except NotImplementedError as error:
        raise ValueError(
            "MATLAB v7.3/HDF5 files are not supported. Save with MATLAB's -v7 option and try again."
        ) from error


def _heights(metadata: dict[str, Any]) -> dict[str, float]:
    result = {}
    for source, target in (("sensor_height_m", "sensorHeightM"), ("height_m", "heightM")):
        value = metadata.get(source)
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            result[target] = float(value)
    return result


def _inspect_cwa(path: Path, configuration: dict[str, Any]) -> dict[str, Any]:
    metadata = _participant_metadata(configuration, {})
    timezone = configuration.get("timezone")
    if not isinstance(timezone, str) or not timezone:
        raise ValueError("The sensor clock synchronization timezone is required.")
    condition = _measurement_condition(configuration, is_cwa=True)
    dataset = AX6Dataset(
        path,
        participant_metadata=metadata,
        recording_metadata={"measurement_condition": condition},
        tz=timezone,
    )
    header = dataset.cwa_header_
    start, end = header["start_from_data_raw"], header["end_from_data_raw"]
    fs = dataset.sampling_rate_hz
    bounds = dataset.index.iloc[0]
    dataset = dataset.clone().set_params(
        subset_index=None,
        splitter=pd.DataFrame(
            {
                "recording": ["inspection"],
                "start_time": [bounds.start_time],
                "end_time": [min(bounds.start_time + pd.Timedelta(seconds=1), bounds.end_time)],
            }
        ),
    )
    channels = list(dataset.data_ss.columns)
    identifier = uuid.uuid4().hex
    description = {
        "id": identifier,
        "sourceFormat": "cwa",
        "fileName": path.name,
        "label": path.name,
        "testName": [path.stem],
        "samples": None,
        "samplingRateHz": fs,
        "durationSeconds": (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() + 1 / fs,
        "channels": channels,
        "sensorPosition": "LowerBack",
        "metadata": _heights(metadata),
        "cwa": {
            "startTimeRaw": start,
            "endTimeRaw": end,
            "hasGyroscope": all(f"gyr_{axis}" in channels for axis in "xyz"),
            "clockTimezoneRequired": True,
        },
        "warnings": [
            "CWA inspection uses AX6Dataset to read boundary metadata and a bounded initial window. "
            "Samples are counted when the selected window is loaded.",
            "CWA analysis assumes a LowerBack sensor in the expected sensor frame. Verify placement and orientation.",
            "Choose the timezone of the computer that synchronized the sensor. The last configuration write is "
            "assumed to be that synchronization; the offset then stays fixed across daylight-saving changes.",
        ],
    }
    _RECORDINGS[identifier] = {
        "path": path,
        "description": description,
        "participant_metadata": metadata,
        "measurement_condition": condition,
    }
    return description


def _cwa_selection(recording_id: str) -> dict[str, Any]:
    selected = _RECORDINGS.get(recording_id)
    if selected is None or selected["description"].get("sourceFormat") != "cwa":
        raise ValueError("Select a CWA recording from the currently loaded files.")
    return selected


def cwa_day_windows(recording_id: str, timezone: str) -> dict[str, Any]:
    """Enumerate local calendar days without decoding sensor measurements."""
    selected = _cwa_selection(recording_id)
    if not timezone:
        raise ValueError("The sensor clock synchronization timezone is required.")
    dataset = AX6Dataset(
        selected["path"],
        participant_metadata=selected["participant_metadata"],
        recording_metadata={"measurement_condition": selected["measurement_condition"]},
        tz=timezone,
        output_timezone="local",
        splitter=split_by_local_days,
    )
    return {"windows": _day_descriptions(dataset, timezone), "timezone": timezone}


def _day_descriptions(dataset: AX6Dataset, timezone: str) -> list[dict[str, Any]]:
    index = dataset.index
    origin = dataset[0].cwa_header_["start_from_data"]
    windows = [
        {
            "index": i,
            "label": row.start_time.tz_convert(timezone).strftime("%Y-%m-%d"),
            "startTime": row.start_time.tz_convert(timezone).isoformat(),
            "endTime": row.end_time.tz_convert(timezone).isoformat(),
            "startSeconds": (row.start_time - origin).total_seconds(),
            "durationSeconds": (row.end_time - row.start_time).total_seconds(),
        }
        for i, row in enumerate(index.itertuples())
    ]
    return windows


def _cwa_dataset(selected: dict[str, Any], options: dict[str, Any], metadata: dict[str, Any], condition: str):
    if not selected["description"]["cwa"]["hasGyroscope"]:
        raise ValueError(
            "Both full presets require gyroscope channels. This CWA recording has none in its initial window."
        )
    day, window = options.get("cwaDay"), options.get("cwaWindow")
    if (day is None) == (window is None):
        raise ValueError("Choose one CWA calendar day or one bounded time window.")
    selection = day if day is not None else window
    timezone = selection.get("timezone")
    if not isinstance(timezone, str) or not timezone:
        raise ValueError("The sensor clock synchronization timezone is required.")
    if day is None:
        start, duration = window.get("startSeconds"), window.get("durationSeconds")
        if (
            not isinstance(start, (int, float))
            or not math.isfinite(start)
            or start < 0
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or not 0 < duration <= 3600
        ):
            raise ValueError("A manual CWA window needs a nonnegative start and duration between 0 and 3600 seconds.")

    dataset = AX6Dataset(
        selected["path"],
        participant_metadata=metadata,
        recording_metadata={"measurement_condition": condition},
        tz=timezone,
        output_timezone="utc",
        splitter=split_by_local_days if day is not None else None,
    )
    if day is None:
        bounds = dataset.index.iloc[0]
        start_time = bounds.start_time + pd.Timedelta(seconds=start)
        end_time = start_time + pd.Timedelta(seconds=duration)
        if start_time >= bounds.end_time or end_time > bounds.end_time:
            raise ValueError("The selected CWA window extends beyond the recording.")
        dataset = dataset.clone().set_params(
            subset_index=None,
            splitter=pd.DataFrame({"recording": ["selected"], "start_time": [start_time], "end_time": [end_time]}),
        )
    else:
        day_index = day.get("index")
        if type(day_index) is not int or not 0 <= day_index < len(dataset):
            raise ValueError("Choose an available CWA calendar day.")
        dataset = dataset[day_index]
    return dataset


def inspect_files(paths: list[str], configuration: dict[str, Any]) -> dict[str, Any]:
    """Inspect worker-filesystem MAT/CWA paths and retain available recordings.

    CWA data remains on disk until a calendar day or manual window is selected.
    Participant metadata must come from a separate infoForAlgo file or manual configuration.
    One infoForAlgo file can accompany one data file regardless of upload filename.
    When inspecting several data files, metadata files must be paired in distinct
    directories, or supplied manually, to avoid assigning another participant's height.
    Unreadable uploads prevent external pairing because their participant is unknown.
    """
    cancel_cwa_day_batch()
    _RECORDINGS.clear()
    cohort = configuration.get("cohort")
    if cohort not in ("HA", "COPD", "CHF", "PD", "MS", "PFF"):
        raise ValueError("Select a participant cohort before building the dataset.")
    errors = []
    parsed = []
    metadata_files = {}
    recordings = []
    cwa_paths = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            if path.suffix.lower() == ".cwa":
                recordings.append(_inspect_cwa(path, configuration))
                cwa_paths.append(path)
                continue
            variables = _mat_variables(path)
            if not variables.intersection({"data", "infoForAlgo"}):
                raise ValueError("This MAT file has no Mobilise-D 'data' or 'infoForAlgo' variable.")
            if "infoForAlgo" in variables and "data" not in variables:
                metadata_files[path] = load_mobilised_participant_metadata_file(path)
            parsed.append((path, variables))
        except Exception as error:  # noqa: BLE001 - untyped upload/pipeline errors cross the browser boundary.
            if _release_exception_frames(error):
                raise
            errors.append(
                {
                    "fileName": path.name,
                    "code": "unsupported_cwa_format" if path.suffix.lower() == ".cwa" else "unsupported_mat_format",
                    "message": str(error),
                }
            )
    data_files = [path for path, variables in parsed if "data" in variables]
    unreadable_uploads = bool(errors)
    for path in data_files:
        try:
            file_warnings = []
            companion = None
            if companion is None and unreadable_uploads and metadata_files:
                file_warnings.append(
                    "Some uploaded files could not be read. Companion participant metadata cannot be paired safely. "
                    "Enter heights manually."
                )
            elif companion is None:
                candidates = [key for key in metadata_files if key.parent == path.parent]
                data_in_directory = sum(key.parent == path.parent for key in [*data_files, *cwa_paths])
                if len(candidates) == 1 and data_in_directory == 1:
                    companion = candidates[0]
                elif len(data_files) + len(cwa_paths) == len(metadata_files) == 1:
                    companion = next(iter(metadata_files))
                elif candidates:
                    file_warnings.append(
                        "Participant metadata is ambiguous for multiple data files. Enter heights manually."
                    )
            metadata_source = companion if companion is not None else _participant_metadata(configuration, {})
            dataset = GenericMobilisedDataset(
                path,
                test_level_names=None,
                participant_metadata_override=metadata_source,
                measurement_condition=_measurement_condition(configuration, is_cwa=False),
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                datapoints = sorted(dataset, key=lambda dp: ("Test11" not in dp.group_label, tuple(dp.group_label)))
            file_warnings.extend(str(item.message) for item in caught)
            for datapoint in datapoints:
                test_name = tuple(datapoint.group_label)
                frame = datapoint.data_ss
                if frame is None or not all(column in frame for column in _REQUIRED_CHANNELS):
                    raise ValueError("A LowerBack sensor with all three acceleration and gyroscope axes is required.")
                fs = datapoint.sampling_rate_hz
                if fs is None or not math.isfinite(fs) or fs <= 0 or frame.empty:
                    raise ValueError(
                        "The recording requires a positive sampling rate and nonempty LowerBack sensor data."
                    )
                if companion is not None:
                    info = metadata_files[companion].get(test_name[0], {})
                    heights = {
                        target: info[source] / 100
                        for source, target in (("Height", "heightM"), ("SensorHeight", "sensorHeightM"))
                        if isinstance(info.get(source), (int, float))
                    }
                else:
                    heights = _heights(datapoint.participant_metadata)
                metadata = _participant_metadata(configuration, heights)
                datapoint = datapoint.clone().set_params(participant_metadata_override=metadata)
                identifier = uuid.uuid4().hex
                description = {
                    "id": identifier,
                    "sourceFormat": "mat",
                    "fileName": path.name,
                    "label": " / ".join(test_name),
                    "testName": list(test_name),
                    "datasetIndex": dict(zip(datapoint.index.columns, test_name, strict=True)),
                    "samples": len(frame),
                    "samplingRateHz": float(fs),
                    "durationSeconds": len(frame) / fs,
                    "channels": list(frame.columns),
                    "sensorPosition": "LowerBack",
                    "metadata": _heights(metadata),
                    "warnings": file_warnings,
                }
                _RECORDINGS[identifier] = {"dataset": datapoint, "description": description}
                recordings.append(description)
                del frame
        except Exception as error:  # noqa: BLE001 - untyped upload/pipeline errors cross the browser boundary.
            if _release_exception_frames(error):
                raise
            errors.append({"fileName": path.name, "code": "unsupported_recording", "message": str(error)})
    return {
        "recordings": recordings,
        "errors": errors,
        "warnings": ["Only participant metadata was loaded. Add the accompanying data MAT file."]
        if not recordings and not data_files and metadata_files
        else [],
    }


def _table(frame: pd.DataFrame) -> dict[str, Any]:
    flat = frame.drop(columns=["rule_obj"], errors="ignore").reset_index()
    columns = [" / ".join(map(str, name)) if isinstance(name, tuple) else str(name) for name in flat.columns]
    return {
        "columns": columns,
        "rows": json.loads(flat.to_json(orient="values", date_format="iso", double_precision=15)),
    }


def _height(options: dict[str, Any], key: str, metadata: dict[str, Any], fallback: str) -> float:
    value = options.get(key, metadata.get(fallback))
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        label = "Sensor height" if key == "sensorHeightM" else "Participant height"
        raise ValueError(f"{label} is required in metres. Enter a positive value.")
    return float(value)


def _participant_metadata(options: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    cohort = options.get("cohort")
    if cohort not in ("HA", "COPD", "CHF", "PD", "MS", "PFF"):
        raise ValueError("Select a participant cohort: HA, COPD, CHF, PD, MS or PFF.")
    return {
        "sensor_height_m": _height(options, "sensorHeightM", fallback, "sensorHeightM"),
        "height_m": _height(options, "participantHeightM", fallback, "heightM"),
        "cohort": cohort,
    }


def _measurement_condition(options: dict[str, Any], *, is_cwa: bool) -> str:
    condition = options.get("measurementCondition", "free_living" if is_cwa else "laboratory")
    if condition not in ("laboratory", "free_living"):
        raise ValueError("Select laboratory or free_living as the measurement condition.")
    return condition


def _analysis_configuration(description: dict[str, Any], options: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    preset = options.get("preset")
    if preset not in ("healthy", "impaired", "auto"):
        raise ValueError("Select the healthy, impaired or auto pipeline preset.")
    return (
        preset,
        _participant_metadata(options, description["metadata"]),
        _measurement_condition(options, is_cwa=description.get("sourceFormat") == "cwa"),
    )


def analyze_recording(recording_id: str, options: dict[str, Any]) -> dict[str, Any]:
    """Run the selected full preset with explicitly supplied cohort and heights."""
    if recording_id not in _RECORDINGS:
        raise ValueError("Select a recording from the currently loaded files before running the pipeline.")
    selected = _RECORDINGS[recording_id]
    description = selected["description"]
    preset, metadata, condition = _analysis_configuration(description, options)
    is_cwa = description.get("sourceFormat") == "cwa"
    if is_cwa:
        dataset = _cwa_dataset(selected, options, metadata, condition)
    else:
        dataset = (
            selected["dataset"]
            .clone()
            .set_params(
                participant_metadata_override=metadata,
                measurement_condition=condition,
            )
        )
    return _analyze_dataset(recording_id, description, dataset, preset)


def _analyze_dataset(recording_id: str, description: dict[str, Any], dataset: Any, preset: str) -> dict[str, Any]:
    is_cwa = description.get("sourceFormat") == "cwa"
    if preset == "auto":
        pipeline = MobilisedPipelineUniversal(
            pipelines=[
                ("healthy", MobilisedPipelineHealthy(retain_intermediate_results=False)),
                ("impaired", MobilisedPipelineImpaired(retain_intermediate_results=False)),
            ]
        )
    else:
        pipeline_class = MobilisedPipelineHealthy if preset == "healthy" else MobilisedPipelineImpaired
        pipeline = pipeline_class(retain_intermediate_results=False)
    started = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if is_cwa:
                frame = dataset.data_ss
                if not all(column in frame for column in _REQUIRED_CHANNELS):
                    raise ValueError("The selected CWA day/window does not contain all required gyroscope channels.")
                sample_count = len(frame)
                del frame
            else:
                sample_count = description["samples"]
            pipeline.run(dataset)
    except Exception as error:  # noqa: BLE001 - untyped upload/pipeline errors cross the browser boundary.
        raise ValueError(f"The {preset} pipeline could not process this recording: {error}") from error
    elapsed = time.perf_counter() - started
    attributes = {
        "gait_sequences": "gs_list_",
        "initial_contacts": "raw_ic_list_",
        "turns": "raw_turn_list_",
        "per_second_parameters": "raw_per_sec_parameters_",
        "raw_per_stride_parameters": "raw_per_stride_parameters_",
        "per_stride_parameters": "per_stride_parameters_",
        "walking_bouts": "per_wb_parameters_",
        "aggregated_parameters": "aggregated_parameters_",
    }
    tables = {
        name: _table(getattr(pipeline, attribute))
        for name, attribute in attributes.items()
        if isinstance(getattr(pipeline, attribute, None), pd.DataFrame)
    }
    summary = {
        "samples": sample_count,
        "durationSeconds": sample_count / dataset.sampling_rate_hz,
        "samplingRateHz": description["samplingRateHz"],
        "gaitSequences": len(pipeline.gs_list_),
        "initialContacts": len(pipeline.raw_ic_list_),
        "walkingBouts": len(pipeline.per_wb_parameters_),
        "strides": len(pipeline.per_stride_parameters_),
        "processingSeconds": elapsed,
    }
    return {
        "recordingId": recording_id,
        "preset": getattr(pipeline, "pipeline_name_", preset),
        "summary": summary,
        "tables": tables,
        "warnings": list(dict.fromkeys([*description["warnings"], *(str(item.message) for item in caught)])),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "mobgap",
                "numpy",
                "scipy",
                "pandas",
                "numba",
                "PyWavelets",
                "xxhash",
                "scikit-learn",
                "tpcp",
                *(("cwa_reader_rs",) if is_cwa else ()),
            )
        },
    }


def _release_exception_frames(error: BaseException) -> bool:
    """Release array-bearing tracebacks and classify chained memory failures."""
    fatal = False
    while error is not None:
        fatal |= isinstance(error, MemoryError)
        if error.__traceback__ is not None:
            traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return fatal


def call_json(operation: Callable[[], Any]) -> str:
    """Serialize one API response without publishing array-bearing exceptions to IPython."""
    try:
        return json.dumps({"ok": True, "result": operation()}, allow_nan=False)
    except Exception as error:  # noqa: BLE001 - Python/JavaScript IPC must retain fatal memory classification.
        message = str(error)
        fatal = _release_exception_frames(error)
        gc.collect()
        return json.dumps({"ok": False, "error": {"message": message, "fatal": fatal}}, allow_nan=False)


def _iterate_cwa_days(recording_id, description, dataset, preset, windows, indices):
    # Iterate one AX6Dataset; each datapoint is one complete calendar day.
    for index, datapoint in enumerate(dataset):
        if index not in indices:
            continue
        packet = {"day": windows[index]}
        try:
            packet["result"] = _analyze_dataset(recording_id, description, datapoint, preset)
        except Exception as error:  # noqa: BLE001 - return day failures without IPython retaining their tracebacks.
            packet["error"] = str(error)
            if _release_exception_frames(error):
                packet["fatal"] = True
        finally:
            del datapoint
            gc.collect()
        yield packet
        if packet.get("fatal"):
            return


def cancel_cwa_day_batch() -> None:
    """Close the active generator between days; it holds no pipeline instance."""
    global _DAY_BATCH
    if _DAY_BATCH is not None:
        _DAY_BATCH.close()
        _DAY_BATCH = None
        gc.collect()


def start_cwa_day_batch(
    recording_id: str, options: dict[str, Any], day_indices: list[int] | None = None
) -> dict[str, Any]:
    """Create one lazy AX6Dataset and one Python loop for selected calendar days."""
    global _DAY_BATCH
    cancel_cwa_day_batch()
    selected = _cwa_selection(recording_id)
    description = selected["description"]
    preset, metadata, condition = _analysis_configuration(description, options)
    timezone = options.get("timezone")
    if not isinstance(timezone, str) or not timezone:
        raise ValueError("The sensor clock synchronization timezone is required.")
    dataset = AX6Dataset(
        selected["path"],
        participant_metadata=metadata,
        recording_metadata={"measurement_condition": condition},
        tz=timezone,
        output_timezone="utc",
        splitter=split_by_local_days,
    )
    windows = _day_descriptions(dataset, timezone)
    indices = list(range(len(windows))) if day_indices is None else sorted(set(day_indices))
    if not indices or any(type(index) is not int or not 0 <= index < len(windows) for index in indices):
        raise ValueError("Choose at least one available CWA calendar day.")
    _DAY_BATCH = _iterate_cwa_days(recording_id, description, dataset, preset, windows, set(indices))
    return {"windows": [windows[index] for index in indices], "totalDays": len(indices)}


def next_cwa_day() -> dict[str, Any]:
    """Advance the Python dataset loop once and return only JSON-safe state."""
    if _DAY_BATCH is None:
        return {"done": True}
    try:
        packet = next(_DAY_BATCH)
    except StopIteration:
        cancel_cwa_day_batch()
        return {"done": True}
    if packet.get("fatal"):
        cancel_cwa_day_batch()
        return {"done": True, "packet": packet}
    return {"done": False, "packet": packet}
