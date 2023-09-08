import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple, Optional, Union

import numpy as np
import pandas as pd
import scipy.io as sio


class MobilisedMetadata(NamedTuple):
    """Metadata of each individual test/recording.

    Parameters
    ----------
    start_date_time_iso
        The start date and time of the test in ISO format.
    time_zone
        The time zone of the test (e.g. "Europe/Berlin").
    sampling_rate_hz
        The sampling rate of the IMU data in Hz.
        None, if no IMU data is available or loaded.
    reference_sampling_rate_hz
        The sampling rate of the reference data in Hz.
        None, if no reference data is available or loaded.

    """
    start_date_time_iso: str
    time_zone: str
    sampling_rate_hz: Optional[float]
    reference_sampling_rate_hz: Optional[float]


class MobilisedTestData(NamedTuple):
    imu_data: Optional[dict[str, pd.DataFrame]]
    # TODO: Update Any typing once I understand the data better.
    reference_parameters: Optional[dict[Literal["wb", "lwb"], Any]]
    metadata: MobilisedMetadata


def load_mobilised_matlab_format(
    path: Union[Path, str],
    *,
    raw_data_sensor: Literal["SU", "INDIP", "INDIP2"] = "SU",
    reference_system: Optional[Literal["INDIP", "Stereophoto"]] = None,
    sensor_positions: Sequence[str] = ("LowerBack",),
    sensor_types: Sequence[Literal["acc", "gyr", "mag", "bar"]] = ("acc", "gyr"),
) -> dict[tuple[str, ...], MobilisedTestData]:
    """Load a single data.mat file formatted according to the Mobilise-D guidelines.

    Parameters
    ----------
    path
        Path to the data.mat file.
    raw_data_sensor
        Which sensor to load the raw data for. One of "SU", "INDIP", "INDIP2".
        SU is usually the "normal" lower back sensor.
        INDIP and INDIP2 are only available under special circumstances for the Mobilise-D TVS data.
    reference_system
        When specified, reference gait parameters are loaded using the specified reference system.
    sensor_positions
        Which sensor positions to load the raw data for.
        For "SU", only "LowerBack" is available, but for other sensors, more positions might be available.
        If a sensor position is not available, an error is raised.
    sensor_types
        Which sensor types to load the raw data for.
        This can be used to reduce the amount of data loaded, if only e.g. acc and gyr data is required.
        Some sensors might only have a subset of the available sensor types.
        If a sensor type is not available, it is ignored.

    Returns
    -------
    data_per_test
        A dictionary mapping the test names to the loaded data.
        The name of each test is a tuple of strings, where each string is a level of the test name hierarchy (e.g.
        ("TimeMeasure1", "Test1", "Trial1")).
        The number of levels can vary between datasets.
        The data is returned as a :class:`~MobilisedTestData` named-tuple, which contains the raw data, (optional)
        reference parameters and metadata.

    Notes
    -----
    This data loader does not cover the entire Mobilise-D data format spec.
    We focus on the loading of the raw single-sensor data and the reference parameters.
    Further, we don't support data files with sensors with different sampling rates or data files that don't specify
    a start time per trial (but only a start time per test).

    """
    if raw_data_sensor is None and reference_system is None:
        raise ValueError(
            "At least one of raw_data_sensor and reference_system must be set. Otherwise no data is loaded."
        )

    raw_data_sensor = ("SU_" + raw_data_sensor) if raw_data_sensor != "SU" else "SU"

    data = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False, mat_dtype=True)
    data_per_test = _parse_until_test_level(data["data"], (raw_data_sensor, "Standards"))
    return {
        test_name: _process_test_data(
            test_data,
            test_name,
            raw_data_sensor=raw_data_sensor,
            reference_system=reference_system,
            sensor_positions=sensor_positions,
            sensor_types=sensor_types,
        )
        for test_name, test_data in data_per_test
    }


def _parse_until_test_level(
    data: sio.matlab.mat_struct, test_level_marker: Sequence[str], _parent_key: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], sio.matlab.mat_struct]]:
    # If one of the test level markers is in the field names, we reached the level of a test.
    if any(marker in data._fieldnames for marker in test_level_marker):
        yield _parent_key, data
        return  # We don't need to parse any further

    for key in data._fieldnames:
        _local_parent_key = (*_parent_key, key)
        val = getattr(data, key)
        if isinstance(val, sio.matlab.mat_struct):
            yield from _parse_until_test_level(val, test_level_marker, _parent_key=_local_parent_key)
        else:
            warnings.warn(
                f"Encountered unexpected data type {type(val)} at key {_local_parent_key} before reaching the "
                "test level. "
                "This might indicate a malformed data file. "
                "Ignoring the key for now.",
                stacklevel=2,
            )


def _process_test_data(  # noqa: PLR0912
    test_data: sio.matlab.mat_struct,
    test_name: tuple[str, ...],
    *,
    raw_data_sensor: Optional[str],
    reference_system: Optional[str],
    sensor_positions: Sequence[str],
    sensor_types: Sequence[Literal["acc", "gyr", "mag", "bar"]],
) -> MobilisedTestData:
    meta_data = {}

    try:
        meta_data["start_date_time_iso"] = test_data.StartDateTime
    except AttributeError as e:
        raise ValueError(f"Start time information is missing from the data file for test {test_name}.") from e

    try:
        meta_data["time_zone"] = test_data.TimeZone
    except AttributeError as e:
        raise ValueError(f"Time zone information is missing from the data file for test {test_name}.") from e

    if raw_data_sensor:
        all_sensor_data = getattr(test_data, raw_data_sensor)
        sampling_rates: dict[str, float] = {}

        all_imu_data = {}
        for sensor_pos in sensor_positions:
            try:
                raw_data = getattr(all_sensor_data, sensor_pos)
            except AttributeError as e:
                raise ValueError(f"Sensor position {sensor_pos} is not available for test {test_name}.") from e

            all_imu_data[sensor_pos] = _parse_single_sensor_data(raw_data, sensor_types)
            sampling_rates_obj = raw_data.Fs
            sampling_rates.update(
                {f"{sensor_pos}_{k}": getattr(sampling_rates_obj, k) for k in sampling_rates_obj._fieldnames}
            )

        # In the data files the sampling rate for each sensor type is reported individually.
        # But in reality, we expect them all to have the same sampling rate.
        # We check that here to simplify the return data structure.
        sampling_rate_values = set(sampling_rates.values())
        if len(sampling_rate_values) != 1:
            raise ValueError(
                f"Expected all sensors across all positions to have the same sampling rate, but found {sampling_rates}"
            )
        meta_data["sampling_rate_hz"] = sampling_rate_values.pop()
    else:
        all_imu_data = None
        meta_data["sampling_rate_hz"] = None

    # In many cases, reference data is only available for a subset of the tests.
    # Hence, we handle the case where the reference data is missing and just return None for this test.
    if reference_system and (reference_data_mat := getattr(test_data.Standards, reference_system, None)):
        meta_data["reference_sampling_rate_hz"] = reference_data_mat.Fs
        # For the supported reference systems, we always get parameters on the level of MircoWB and
        # ContinuousWalkingPeriod.
        # However, this naming was changed at some point (MircoWB -> LWB and ContinuousWalkingPeriod-> WB).
        # TODO: I don't know, if any newer data files actually use the new naming.
        #       We just check for the old names and rename them to the new ones.
        reference_data = {}
        try:
            reference_data["lwb"] = _parse_reference_parameters(reference_data_mat.MicroWB)
            reference_data["wb"] = _parse_reference_parameters(reference_data_mat.ContinuousWalkingPeriod)
        except AttributeError as e:
            raise ValueError(
                f"Reference data using the reference system {reference_system} for test {test_name} is missing results "
                "for either LWBs/MicroWBs or WBs/ContinuousWalkingPeriods or parsing of the respective data failed."
            ) from e
    else:
        reference_data = None
        meta_data["reference_sampling_rate_hz"] = None

    return MobilisedTestData(imu_data=all_imu_data, reference_parameters=reference_data, metadata=MobilisedMetadata(**meta_data))


def _parse_single_sensor_data(
    sensor_data: sio.matlab.mat_struct, sensor_types: Sequence[Literal["acc", "gyr", "mag", "bar"]]
) -> pd.DataFrame:
    parsed_data = []
    for sensor_type in sensor_types:
        if (sensor_type_mat := sensor_type.capitalize()) in sensor_data._fieldnames:
            if sensor_type == "bar":
                column_names = [f"{sensor_type}"]
            else:
                column_names = [f"{sensor_type}_{axis}" for axis in ("x", "y", "z")]
            parsed_data.append(pd.DataFrame(getattr(sensor_data, sensor_type_mat), columns=column_names))
    parsed_data = pd.concat(parsed_data, axis=1)
    # Some sensors provide realtime timestamps.
    # If they are available, we load them as the index.
    if "Timestamp" in sensor_data._fieldnames:
        parsed_data.index = pd.DatetimeIndex(pd.to_datetime(sensor_data.Timestamp, unit="s", utc=True), name="time")
    else:
        parsed_data.index.name = "samples"

    return parsed_data


def _parse_reference_parameters(
    reference_data: Union[sio.matlab.mat_struct, list[sio.matlab.mat_struct]],
) -> list[dict[str, Union[str, float, int, np.ndarray]]]:
    # For now reference data is either a list of structs or a single struct.
    # Each struct represents one walking bout
    # Each struct has various fields that either contain a single value or a list of values (np.arrays).
    # We perform the conversion in a way that we always return a list of dicts.
    if isinstance(reference_data, sio.matlab.mat_struct):
        reference_data = [reference_data]
    return [{k: getattr(wb_data, k) for k in wb_data._fieldnames} for wb_data in reference_data]
