import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Literal,
    NamedTuple,
    Optional,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import joblib
import numpy as np
import pandas as pd
import scipy.io as sio
from pandas._libs.missing import NAType
from tpcp.caching import hybrid_cache
from tqdm.auto import tqdm

import mobgap
from mobgap._docutils import make_filldoc
from mobgap.consts import GRAV_MS2
from mobgap.data.base import (
    BaseGaitDatasetWithReference,
    ParticipantMetadata,
    RecordingMetadata,
    ReferenceData,
    base_gait_dataset_docfiller_dict,
)
from mobgap.utils.conversions import as_samples

T = TypeVar("T")

PathLike = TypeVar("PathLike", str, Path)

matlab_dataset_docfiller = make_filldoc(
    {
        **base_gait_dataset_docfiller_dict,
        "file_loader_args": """
    raw_data_sensor
        Which sensor to load the raw data for. One of "SU", "INDIP", "INDIP2".
        SU is usually the "normal" lower back sensor.
        INDIP and INDIP2 are only available under special circumstances for the Mobilise-D TVS data.
        Note, that we don't support loading multiple sensors at once.
    reference_system
        When specified, reference gait parameters are loaded using the specified reference system.
    sensor_positions
        Which sensor positions to load the raw data for.
        For "SU", only "LowerBack" is available, but for other sensors, more positions might be available.
        If a sensor position is not available, an error is raised.
    single_sensor_position
        The sensor position that is considered the "single sensor".
        This is the sensor that you expect to be the input to all pipelines and algorithms.
        For most Mobilise-d datasets, this should be kept at "LowerBack".
        But, we support using other sensors as well.
    sensor_types
        Which sensor types to load the raw data for.
        This can be used to reduce the amount of data loaded, if only e.g. acc and gyr data is required.
        Some sensors might only have a subset of the available sensor types.
        If a sensor type is not available, it is ignored.
    missing_sensor_error_type
        Whether to throw an error ("raise"), a warning ("warn") or ignore ("ignore") when a sensor is missing.
        In all three cases, the trial is still included in the index, but the imu-data is not available.
        If you want to skip the trial entirely, set this to "skip".
        Then the trial will not appear in the index at all.
        Note, that "skip" will skip the trial, if ANY sensor position specified is missing.
        Specifying, "skip" will only effect the initial data loading.
        Changing the value after you already created a subset of the data has no effect.
    missing_reference_error_type
        Whether to throw an error ("raise"), a warning ("warn") or ignore ("ignore") when reference data is missing.
        If you want to skip the trial entirely when the reference data is not available, set this to "skip".
        Specifying, "skip" will only effect the initial data loading.
        Changing the value after you already created a subset of the data has no effect.
    """,
        "general_dataset_args": """
    reference_para_level
        Whether to provide "wb" (walking bout) or "lwb" (level-walking bout) reference when loading
        ``reference_parameters_``.
        ``raw_reference_parameters_`` will always contain both in an unformatted way.
    """
        + base_gait_dataset_docfiller_dict["general_dataset_args"],
        "dataset_data_attrs": base_gait_dataset_docfiller_dict["common_dataset_data_attrs"]
        + base_gait_dataset_docfiller_dict["common_dataset_reference_attrs"]
        + """
    participant_metadata_as_df
        The participant metadata as a DataFrame.
        This contains the same information as ``participant_metadata``, but the property can be accessed even when the
        dataset still contains multiple participants.
        It contains one row per participant and are all columns of the index, that are required to uniquely identify a
        single measurement.
    recording_metadata_as_df
        The recording metadata as a DataFrame.
        This contains the same information as ``recording_metadata``, but the property can be accessed even when the
        dataset still contains multiple participants.
        It contains one row for each row in the dataset index.
    raw_reference_parameters_
        The raw reference parameters (if available).
        Check other attributes with a trailing underscore for the reference parameters converted into a more
        standardized format.
    metadata
        The metadata of the selected test.
    """
        + base_gait_dataset_docfiller_dict["dataset_classvars"],
        "dataset_see_also": base_gait_dataset_docfiller_dict["dataset_see_also"]
        + """
    load_mobilised_matlab_format
    """,
    }
)


class MobilisedParticipantMetadata(ParticipantMetadata):
    """Participant metadata for the Mobilise-D dataset.

    This is a subclass of :class:`~ParticipantMetadata` and adds some additional fields that are specific to the
    Mobilise-D dataset.

    Attributes
    ----------
    height_m
        The height of the participant in meters.
    sensor_height_m
        The height of the lower back sensor in meters.
    cohort
        The cohort of the participant.
        One of:
        - "HA": Healthy Adult
        - "MS": Multiple Sclerosis
        - "PD": Parkinson's Disease
        - "COPD": Chronic Obstructive Pulmonary Disease
        - "CHF": Chronic Heart Failure
        - "PFF": Primary Frailty Fracture
    foot_length_cm
        The foot size in cm.
    weight_kg
        The weight of the participant in kg.
    handedness
        The handedness of the participant.
        Either "left" or "right".
    indip_data_used
        Whether all INDIP data was used, or just partial data, as some sensors failed.
    sensor_attachment_su
        Where and how the SU was attached
    sensor_type_su
        The type of SU used (usually MM+ or AX6)
    walking_aid_used
        Whether a walking aid was used during the test.

    """

    foot_length_cm: Optional[float]
    weight_kg: Optional[float]
    handedness: Optional[Literal["left", "right"]]
    indip_data_used: Optional[str]
    sensor_attachment_su: Optional[str]
    sensor_type_su: Optional[str]
    walking_aid_used: Optional[bool]


class PartialMobilisedMetadata(TypedDict):
    """Metadata of each individual test/recording.

    Attributes
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
    time_zone: Optional[str]
    sampling_rate_hz: Optional[float]
    reference_sampling_rate_hz: Optional[float]


class MobilisedMetadata(PartialMobilisedMetadata, RecordingMetadata):
    """Metadata of each individual test/recording.

    Attributes
    ----------
    measurement_condition
        The measurement condition of the test (e.g. "laboratory" or "free_living").
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
    recording_identifier
        A tuple with the measurement, test, trial name
    """

    recording_identifier: tuple[str, ...]


class MobilisedAvailableData(NamedTuple):
    """Basic metadata that can be used to filter tests based on availability of certain factors.

    Parameters
    ----------
    available_sensors
        The available sensors for the test.
        It provides a list of tuples of the form (``sensor_type``, ``sensor_position``).
    available_reference_systems
        The available reference systems for the test.
        This is a list

    """

    available_sensors: list[tuple[Literal["SU", "INDIP", "INDIP2"], str]]
    available_reference_systems: list[Literal["INDIP", "Stereophoto"]]


class MobilisedTestData(NamedTuple):
    """Data representation of a single test/recording.

    Parameters
    ----------
    imu_data
        The raw IMU data.
        This is a dictionary mapping the sensor position to the loaded data.
        In most cases, only "LowerBack" is available.
    raw_reference_parameters
        The reference parameters (if available).
        This will depend on the reference system used loaded.
        The parameter only represents the data of one system.
        All compliant reference systems, structure the output into `wb` (walking bout) and `lwb` (level-walking bout).
        The actual structure of the data depends on the reference system.
    metadata
        The metadata of the selected test.
        At the moment we only support cases where all imu-sensors have the same sampling rate
    """

    imu_data: Optional[dict[str, pd.DataFrame]]
    raw_reference_parameters: Optional[dict[Literal["wb", "lwb"], Any]]
    metadata: PartialMobilisedMetadata


def load_mobilised_participant_metadata_file(
    path: PathLike,
) -> dict[str, dict[str, Any]]:
    """Load the participant metadata file (usually called infoForAlgo.mat).

    This file contains various metadata about the participant and the measurement setup.
    There should be one file per corresponding data file.

    Parameters
    ----------
    path
        Path to the infoForAlgo.mat file.

    Returns
    -------
    info_for_algo
        A dictionary with two levels.
        The first level corresponds to the first level of the test-names in the corresponding data file (usually the
        TimeMeasure).
        The second level contains the actual metadata.

    """
    info_for_algo = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False, mat_dtype=True)["infoForAlgo"]
    # The first level of the "infoForAlgo" file is the TimeMeasure.
    # This should correspond to the first level of test-names in the corresponding data file.
    return {
        time_measure: _parse_matlab_struct(getattr(info_for_algo, time_measure))
        for time_measure in info_for_algo._fieldnames
    }


def _extract_available_sensor_pos_(
    test_data: sio.matlab.mat_struct,
) -> list[tuple[Literal["SU", "INDIP", "INDIP2"], str]]:
    result = []
    for field in test_data._fieldnames:
        if field == "SU":
            short_field_name = "SU"
        elif field.startswith("SU_"):
            # To be consistent with how we represent the names of the non "SU" sensors, we remove the "SU_" prefix.
            short_field_name = field[3:]
        else:
            continue
        positions = getattr(test_data, field)._fieldnames
        for position in positions:
            result.append((short_field_name, position))
    return result


def _extract_available_reference_systems(test_data: sio.matlab.mat_struct) -> list[Literal["INDIP", "Stereophoto"]]:
    try:
        standards = test_data.Standards._fieldnames
    except AttributeError:
        return []

    # There can be a bunch of different fields here. We are only interested in "INDIP" and "Stereophoto".
    # We simply check if they exist.
    return [field for field in standards if field in ["INDIP", "Stereophoto"]]


@matlab_dataset_docfiller
def load_mobilised_matlab_format(
    path: PathLike,
    *,
    raw_data_sensor: Optional[Literal["SU", "INDIP", "INDIP2"]] = "SU",
    reference_system: Optional[Literal["INDIP", "Stereophoto"]] = None,
    sensor_positions: Sequence[str] = ("LowerBack",),
    sensor_types: Sequence[Literal["acc", "gyr", "mag", "bar"]] = ("acc", "gyr"),
    missing_sensor_error_type: Literal["raise", "warn", "ignore"] = "raise",
    missing_reference_error_type: Literal["raise", "warn", "ignore"] = "ignore",
) -> dict[tuple[str, ...], MobilisedTestData]:
    """Load a single data.mat file formatted according to the Mobilise-D guidelines.

    Parameters
    ----------
    path
        Path to the data.mat file.
    %(file_loader_args)s

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
    data_per_test, available_data_per_test = _load_test_data_without_checks(
        path,
        raw_data_sensor=raw_data_sensor,
        reference_system=reference_system,
        sensor_positions=sensor_positions,
        sensor_types=sensor_types,
    )

    return {
        test_name: test_data
        for test_name, test_data in data_per_test.items()
        if _check_test_data(
            available_data_per_test[test_name],
            raw_data_sensor=raw_data_sensor,
            reference_system=reference_system,
            sensor_positions=sensor_positions,
            missing_sensor_error_type=missing_sensor_error_type,
            missing_reference_error_type=missing_reference_error_type,
            error_context=f"Test: {test_name}, File: {path}",
        )
    }


def _load_test_data_without_checks(
    path: PathLike,
    *,
    raw_data_sensor: Optional[Literal["SU", "INDIP", "INDIP2"]] = "SU",
    reference_system: Optional[Literal["INDIP", "Stereophoto"]] = None,
    sensor_positions: Sequence[str] = ("LowerBack",),
    sensor_types: Sequence[Literal["acc", "gyr", "mag", "bar"]] = ("acc", "gyr"),
) -> tuple[dict[tuple[str, ...], MobilisedTestData], dict[tuple[str, ...], MobilisedAvailableData]]:
    """Like load_mobilised_matlab_format, but does not perform any checks.

    The information of what data is available is simply returned, so that checks can be performed later.
    """
    if raw_data_sensor is None and reference_system is None:
        raise ValueError(
            "At least one of raw_data_sensor and reference_system must be set. Otherwise no data is loaded."
        )

    if raw_data_sensor:
        raw_data_sensor = ("SU_" + raw_data_sensor) if raw_data_sensor != "SU" else "SU"

    sensor_test_level_marker = raw_data_sensor or "SU"

    data = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False, mat_dtype=True)
    data_per_test = _parse_until_test_level(data["data"], (sensor_test_level_marker, "Standards"))

    data_per_test_parsed = {}
    available_data_per_test = {}
    for test_name, test_data in data_per_test:
        data_per_test_parsed[test_name] = _process_test_data(
            test_data,
            raw_data_sensor=raw_data_sensor,
            reference_system=reference_system,
            sensor_positions=sensor_positions,
            sensor_types=sensor_types,
        )
        available_data_per_test[test_name] = MobilisedAvailableData(
            available_sensors=_extract_available_sensor_pos_(test_data),
            available_reference_systems=_extract_available_reference_systems(test_data),
        )

    return data_per_test_parsed, available_data_per_test


def _parse_until_test_level(
    data: sio.matlab.mat_struct,
    test_level_marker: Sequence[str],
    _parent_key: tuple[str, ...] = (),
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


def _process_test_data(  # noqa: C901, PLR0912
    test_data: sio.matlab.mat_struct,
    *,
    raw_data_sensor: Optional[str],
    reference_system: Optional[str],
    sensor_positions: Sequence[str],
    sensor_types: Sequence[Literal["acc", "gyr", "mag", "bar"]],
) -> MobilisedTestData:
    # Note, this function ignores all missing sensor loadings, as we expect the caller to handle this.
    meta_data = {}

    try:
        meta_data["start_date_time_iso"] = test_data.StartDateTime
    except AttributeError:
        # TODO: Make this handling conditional, so that you have to specify, if you assume a start time or not.
        meta_data["start_date_time_iso"] = None
        # raise ValueError(f"Start time information is missing from the data file for test {test_name} in {path}.")
        # from e

    try:
        meta_data["time_zone"] = test_data.TimeZone
    except AttributeError:
        # TODO: Make this handling conditional, so that you have to specify, if you assume a time zone or not.
        meta_data["time_zone"] = None
        # raise ValueError(f"Time zone information is missing from the data file for test {test_name} in {path}.")
        # from e

    if raw_data_sensor:
        all_sensor_data = getattr(test_data, raw_data_sensor, {})
        sampling_rates: dict[str, float] = {}

        all_imu_data = {}
        for sensor_pos in sensor_positions:
            try:
                raw_data = getattr(all_sensor_data, sensor_pos)
            except AttributeError:
                continue
                # We ignore the error here, as we handle missing sensors later.
            else:
                all_imu_data[sensor_pos] = _parse_single_sensor_data(raw_data, sensor_types)
                sampling_rates_obj = raw_data.Fs
                sampling_rates.update(
                    {f"{sensor_pos}_{k}": getattr(sampling_rates_obj, k) for k in sampling_rates_obj._fieldnames}
                )

        # In the data files the sampling rate for each sensor type is reported individually.
        # But in reality, we expect them all to have the same sampling rate.
        # We check that here to simplify the return data structure.
        # If no sampling rate has been extracted, return None
        sampling_rate_values = set(sampling_rates.values())

        if len(sampling_rate_values) > 1:
            raise ValueError(
                f"Expected all sensors across all positions to have the same sampling rate, but found {sampling_rates}"
            )

        if sampling_rate_values:
            meta_data["sampling_rate_hz"] = sampling_rate_values.pop()
        else:
            meta_data["sampling_rate_hz"] = None

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
        for name, new_name in [("MicroWB", "lwb"), ("ContinuousWalkingPeriod", "wb")]:
            if not hasattr(reference_data_mat, name):
                continue
            reference_data[new_name] = _parse_reference_parameters(getattr(reference_data_mat, name))
    else:
        reference_data = None
        meta_data["reference_sampling_rate_hz"] = None

    return MobilisedTestData(
        imu_data=all_imu_data,
        raw_reference_parameters=reference_data,
        metadata=PartialMobilisedMetadata(**meta_data),
    )


def _check_test_data(
    available_data: MobilisedAvailableData,
    *,
    raw_data_sensor: Optional[Literal["SU", "INDIP", "INDIP2"]],
    reference_system: Optional[Literal["INDIP", "Stereophoto"]],
    sensor_positions: Sequence[str],
    missing_sensor_error_type: Literal["raise", "warn", "ignore", "skip"],
    missing_reference_error_type: Literal["raise", "warn", "ignore", "skip"],
    error_context: Optional[str] = None,
) -> bool:
    # Return boolean indicates if it should be included, not, if it is valid!
    if raw_data_sensor:
        available_sensors = available_data.available_sensors
        expected_sensors = [(raw_data_sensor, pos) for pos in sensor_positions]
        sensor_data_missing = set(expected_sensors) - set(available_sensors)

        if sensor_data_missing:
            error_context = f"Context: {error_context}\n" if error_context else ""
            error_message = f"{error_context}Expected sensor data for {sensor_data_missing}, but it is missing."
            if missing_sensor_error_type == "raise":
                raise ValueError(error_message)
            if missing_sensor_error_type == "warn":
                warnings.warn(error_message, stacklevel=2)
            elif missing_sensor_error_type == "skip":
                return False

    available_reference_systems = available_data.available_reference_systems
    if reference_system and reference_system not in available_reference_systems:
        error_message = f"Expected reference system {reference_system}, but it is not available."
        if missing_reference_error_type == "raise":
            raise ValueError(error_message)
        if missing_reference_error_type == "warn":
            warnings.warn(error_message, stacklevel=2)
        elif missing_reference_error_type == "skip":
            return False

    return True


def _parse_single_sensor_data(
    sensor_data: sio.matlab.mat_struct,
    sensor_types: Sequence[Literal["acc", "gyr", "mag", "bar"]],
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
    # Convert acc columns to m/s-2
    parsed_data[["acc_x", "acc_y", "acc_z"]] *= GRAV_MS2

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
    return [_parse_matlab_struct(wb_data) for wb_data in reference_data]


def _parse_matlab_struct(struct: sio.matlab.mat_struct) -> dict[str, Any]:
    """Parse a simple matlab struct that only contains simple types (no nested structs or arrays)."""
    return {k: getattr(struct, k) for k in struct._fieldnames}


def _ensure_is_list(value: Any) -> list:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list):
        return [value]
    return value


def _empty_list_to_nan(value: list) -> Union[list, NAType]:
    if not value:
        return pd.NA
    return value


def parse_reference_parameters(  # noqa: C901, PLR0912, PLR0915
    ref_data: list[dict[str, Union[str, float, int, np.ndarray]]],
    *,
    ref_sampling_rate_hz: float,
    data_sampling_rate_hz: float,
    relative_to_wb: bool = False,
    debug_info: str,
    not_expected_fields: Optional[Sequence[str]] = None,
    ignore_expected_warnings: bool = False,
) -> ReferenceData:
    """Parse the reference data (stored per WB) into the per recording data structures used in mobgap.

    .. note :: This expects the reference for only onw of the WB levels as input (i.e. `reference_data["wb"]` not
        `reference_data`).

    This does the following:

    - Using the wb start and end values to build a list of reference gait sequences.
    - Concatenate all initial contacts into a single list.
    - Concatenate all turn parameters into a single list.
    - Concatenate all stride parameters into a single list.
    - All time values are converted from seconds to samples of the single sensor system since the start of the recording
    - Further, we drop all strides and ICs that have a NaN value, as this has no meaning outside the context of the
      reference system.
      However, all strides that have NaNs for other parameters are kept.

    This function is also the place to fix some of the inconsistencies in the reference data format.
    Things that are currently fixed here:

    - The reference start-end values for each stride for the INDIP system are provided in samples, not in seconds.
      This is handled here, and independent of the reference system, we correctly convert all values into samples.
    - Drop duplicate ICs and strides.

    Parameters
    ----------
    ref_data
        The raw reference data for one of the WB levels.
    ref_sampling_rate_hz
        The sampling rate of the reference data in Hz.
        This is required to convert the values that are provided in samples of the reference system into samples of the
        single sensor system.
    data_sampling_rate_hz
        The sampling rate of the raw data in Hz.
        This is used to convert values that are provided in seconds into samples.
    relative_to_wb
        Whether to convert all values to be relative to the start of each individual WB.
        This will of course not affect the WB start and end values, but all other values (events, strides, ...) will be
        converted.
    debug_info
        A string that is shown in all warnings and errors that are raised during the parsing.
    not_expected_fields
        A list of fields that are not expected in the reference data.
        This can be used to skip the parsing of certain fields.
        This is useful, if the reference data is incomplete and certain fields are not available.
        The names of the fields are equivalent to the keys in the ``ReferenceData`` object.
    ignore_expected_warnings
        If True, we will ignore warnings that are expected due to issues with the INDIP reference data.
        This includes occasional duplicate initial contacts, and initial contacts that are placed after the end of the
        walking bout.


    Returns
    -------
    ReferenceData
        A named tuple with the parsed reference data.
        See :class:`~ReferenceData` for details.

    See Also
    --------
    ParsedReferenceData
        The output data structure.

    """
    walking_bouts = []
    ics = []
    turn_paras = []
    stride_paras = []

    def warn(message: str) -> None:
        if ignore_expected_warnings:
            return
        warnings.warn(f"{message}\nThis warning happened at {debug_info}", stacklevel=2)

    wb_df_dtypes = {
        "wb_id": "int64",
        "start": "int64",
        "end": "int64",
        "n_strides": "int64",
        "duration_s": "float64",
        "length_m": "float64",
        "avg_walking_speed_mps": "float64",
        "avg_cadence_spm": "float64",
        "avg_stride_length_m": "float64",
        "termination_reason": "string",
    }

    ic_df_dtypes = {
        "wb_id": "int64",
        "step_id": "int64",
        "ic": "int64",
        "lr_label": pd.CategoricalDtype(categories=["left", "right"]),
    }

    turn_df_dtypes = {
        "wb_id": "int64",
        "turn_id": "int64",
        "start": "int64",
        "end": "int64",
        "duration_s": "float64",
        "angle_deg": "float64",
        "direction": pd.CategoricalDtype(categories=["left", "right"]),
    }

    stride_df_dtypes = {
        "wb_id": "int64",
        "s_id": "int64",
        "start": "int64",
        "end": "int64",
        "duration_s": "float64",
        "length_m": "float64",
        "speed_mps": "float64",
        "stance_time_s": "float64",
        "swing_time_s": "float64",
    }

    def _unify_wb_df(df: pd.DataFrame) -> pd.DataFrame:
        return df.astype(wb_df_dtypes).set_index("wb_id")

    def _unify_ic_df(df: pd.DataFrame) -> pd.DataFrame:
        return df.astype(ic_df_dtypes).set_index(["wb_id", "step_id"])

    def _unify_turn_df(df: pd.DataFrame) -> pd.DataFrame:
        return df.astype(turn_df_dtypes).set_index(["wb_id", "turn_id"])

    def _unify_stride_df(df: pd.DataFrame) -> pd.DataFrame:
        return df.astype(stride_df_dtypes).set_index(["wb_id", "s_id"])

    expect_none = dict.fromkeys(not_expected_fields) if not_expected_fields else {}

    if len(ref_data) == 0:
        return ReferenceData(
            _unify_wb_df(pd.DataFrame(columns=list(wb_df_dtypes.keys()))),
            _unify_ic_df(pd.DataFrame(columns=list(ic_df_dtypes.keys()))),
            _unify_turn_df(pd.DataFrame(columns=list(turn_df_dtypes.keys()))),
            _unify_stride_df(pd.DataFrame(columns=list(stride_df_dtypes.keys()))),
        )._replace(**expect_none)

    for wb_id, wb in enumerate(ref_data):
        parsed_wb = {
            "wb_id": wb_id,
            "start": as_samples(wb["Start"], data_sampling_rate_hz),
            "end": as_samples(wb["End"], data_sampling_rate_hz),
            "n_strides": wb["NumberStrides"],
            "duration_s": wb["Duration"],
            "length_m": wb["Length"],
            "avg_walking_speed_mps": wb["WalkingSpeed"],
            "avg_cadence_spm": wb["Cadence"],
            "avg_stride_length_m": wb["AverageStrideLength"],
            "termination_reason": wb.get("TerminationReason", "Not Specified"),
        }
        walking_bouts.append(parsed_wb)

        if "initial_contacts" not in expect_none:
            ic_vals = _ensure_is_list(wb["InitialContact_Event"])
            ic_vals = pd.DataFrame.from_dict(
                {
                    "wb_id": [wb_id] * len(ic_vals),
                    "step_id": np.arange(0, len(ic_vals)),
                    "ic": ic_vals,
                    "lr_label": _ensure_is_list(wb["InitialContact_LeftRight"]),
                }
            )
            ic_vals["ic"] = (ic_vals["ic"] * data_sampling_rate_hz).round()
            # We dropnas here, as in the TVS version 1.03, ic vals can be NaN, if they come from strides that are not
            # clearly defined in the signal, but we know they should exist.
            ic_vals = ic_vals.dropna(how="any")
            # We also get the correct LR-label for the stride parameters from the ICs.
            ic_duplicate_as_nan = ic_vals.copy()
            # We first drop duplicates. This will get rid of ICs within a single WB that are just duplicated for some
            # reason.
            # Note, that we still allow identical WB across different WBs.
            # This could still be considered a bug in the reference system, but we decided that we can not fix that
            # easily.
            ic_duplicate_as_nan = ic_duplicate_as_nan.drop_duplicates(subset=["ic", "lr_label"])
            # Then we set the LR label of all ICs that are still duplicated to NaN (i.e. same IC, but different LR
            # label).
            ic_duplicate_as_nan.loc[ic_duplicate_as_nan["ic"].duplicated(keep=False), "lr_label"] = pd.NA
            # After setting the LR label to NaN, we drop the duplicates again.
            # As the LR label is now NaN, the rows are considered duplicated.
            ic_duplicate_as_nan = ic_duplicate_as_nan.drop_duplicates(subset=["ic", "lr_label"])
            if ic_duplicate_as_nan["lr_label"].isna().any():
                warn(
                    "There were multiple ICs with the same index value, but different LR labels in WB "
                    f"{wb_id}. "
                    "This is likely an issue with the reference system you should further investigate. "
                    "For now, we set the `lr_label` of the stride corresponding to this IC to Nan and drop the "
                    "duplicate. "
                )
            if ic_duplicate_as_nan["ic"].max() > parsed_wb["end"]:
                warn(
                    f"Some initial contacts in WB {wb_id} are after the end of the walking bout. "
                    "This is a known issue for the TVS data using the INDIP reference system. "
                    "It is caused by a bug, where sometimes the end of the walking bout is not correctly set when"
                    "it is not the last stride that has the last IC, but an earlier stride due to missing ICs."
                )
                ic_duplicate_as_nan = ic_duplicate_as_nan[ic_duplicate_as_nan["ic"] <= parsed_wb["end"]]
            ics.append(ic_duplicate_as_nan.sort_values("ic"))
        else:
            ics.append(pd.DataFrame(columns=list(set(ic_df_dtypes.keys()))))
        if "turn_parameters" not in expect_none:
            turn_starts = _ensure_is_list(wb["Turn_Start"])
            turn_paras.append(
                pd.DataFrame.from_dict(
                    {
                        "wb_id": [wb_id] * len(turn_starts),
                        "turn_id": np.arange(0, len(turn_starts)),
                        "start": turn_starts,
                        "end": _ensure_is_list(wb["Turn_End"]),
                        "duration_s": _ensure_is_list(wb["Turn_Duration"]),
                        "angle_deg": _ensure_is_list(wb["Turn_Angle"]),
                    }
                )
            )
        else:
            turn_paras.append(pd.DataFrame(columns=list(set(turn_df_dtypes.keys()))))

        if "stride_parameters" not in expect_none:
            starts, ends = zip(*_ensure_is_list(wb["Stride_InitialContacts"]))
            stride_paras.append(
                pd.DataFrame.from_dict(
                    {
                        "wb_id": [wb_id] * len(starts),
                        "s_id": np.arange(0, len(starts)),
                        "start": starts,
                        "end": ends,
                        # For some reason, the matlab files contains empty arrays to signal a "missing" value in the
                        # data columns for the Stereo-photo system. We replace them with NaN using `to_numeric`.
                        "duration_s": pd.to_numeric(_empty_list_to_nan(_ensure_is_list(wb["Stride_Duration"]))),
                        "length_m": pd.to_numeric(_empty_list_to_nan(_ensure_is_list(wb["Stride_Length"]))),
                        "speed_mps": pd.to_numeric(_empty_list_to_nan(_ensure_is_list(wb["Stride_Speed"]))),
                        "stance_time_s": pd.to_numeric(_empty_list_to_nan(_ensure_is_list(wb["Stance_Duration"]))),
                        "swing_time_s": pd.to_numeric(_empty_list_to_nan(_ensure_is_list(wb["Swing_Duration"]))),
                    }
                )
            )
        else:
            stride_paras.append(pd.DataFrame(columns=list(set(stride_df_dtypes.keys()))))

    walking_bouts = pd.DataFrame.from_records(walking_bouts)
    # For some reason, the matlab code contains empty arrays to signal a "missing" value in the data
    # columns for the Stereophoto system. We replace them with NaN using `to_numeric`.
    for col in [
        "n_strides",
        "duration_s",
        "n_strides",
        "duration_s",
        "length_m",
        "avg_walking_speed_mps",
        "avg_cadence_spm",
        "avg_stride_length_m",
    ]:
        walking_bouts[col] = pd.to_numeric(walking_bouts[col])

    walking_bouts = walking_bouts.replace(np.array([]), np.nan)
    walking_bouts = _unify_wb_df(walking_bouts)

    ics = pd.concat(ics, ignore_index=True)
    ics_is_na = ics["ic"].isna()
    ics = ics[~ics_is_na].drop_duplicates()
    # make left-right labels lowercase
    ics["lr_label"] = ics["lr_label"].str.lower()
    ics = _unify_ic_df(ics)

    turn_paras = (
        pd.concat(turn_paras, ignore_index=True)
        .assign(direction=lambda df_: np.sign(df_["angle_deg"]))
        .replace({"direction": {1: "left", -1: "right"}})
    )
    turn_paras[["start", "end"]] = (turn_paras[["start", "end"]] * data_sampling_rate_hz).round()
    turn_paras = _unify_turn_df(turn_paras)

    if len(stride_paras) > 0:
        stride_paras = pd.concat(stride_paras, ignore_index=True)
        stride_ics_is_na = stride_paras[["start", "end"]].isna().any(axis=1)
        stride_paras = stride_paras[~stride_ics_is_na]
        # Note: For the INDIP system it seems like start and end are provided in samples already and not in seconds.
        #       I am not sure what the correct behavior is, but we try to handle it to avoid confusion on the user side.
        #       Unfortunately, there is no 100% reliable way to detect this, so we just check if the values are in the
        #       IC list (which seems to be provided in time in all ref systems I have seen).
        #
        # If we assume the values are in samples of the reference system, than we attempt to convert them to samples of
        # the single sensor system.
        # For the INDIP system, that shouldn't matter, as the sampling rates are the same, but hey you can never be too
        # safe.
        assume_stride_paras_in_samples = (
            (stride_paras["start"] * (data_sampling_rate_hz / ref_sampling_rate_hz)).round().astype("int64")
        )
        # ICs are already converted to samples here -> I.e. if they are not all in here, we assume that the stride
        # parameters are also in seconds not in samples.
        if not assume_stride_paras_in_samples.isin(ics["ic"]).all():
            warn("Assuming stride start and end values are provided in seconds and not in samples. ")
            stride_paras[["start", "end"]] = (
                (stride_paras[["start", "end"]] * data_sampling_rate_hz).round().astype("int64")
            )
            # We check again, just to be sure and if they are still not there, we throw an error.
            if not stride_paras["start"].isin(ics["ic"]).all():
                raise ValueError(
                    "There seems to be a mismatch between the provided stride parameters and the provided initial "
                    "contacts of the reference system. "
                    "At least some of the ICs marking the start of a stride are not found in the dedicated IC list. "
                    f"This error happened at {debug_info}",
                )
        else:
            stride_paras[["start", "end"]] = (
                (stride_paras[["start", "end"]] * (data_sampling_rate_hz / ref_sampling_rate_hz))
                .round()
                .astype("int64")
            )

    # Note that we need to drop duplicates again here, as the ics can contain the same IC multiple times in different
    # WBs. This is a quirk of the INDIP system.
    stride_paras["lr_label"] = ics.drop_duplicates().set_index("ic").loc[stride_paras["start"], "lr_label"].to_numpy()
    stride_paras = _unify_stride_df(stride_paras)

    # Due to the way, on how the data is used on matlab side, we need to adjust the indices of all time values.
    # We need to fix 2 things:
    # 1. In Matlab time value were calculated to samples (as done here), but then used as indices in Matlabs 1-based
    #    indexing. To make them correspond to the 0-based indexing in python, we need to subtract 1 from all values.
    # 2. In Matlab slicing is inclusive, but in python it is exclusive. Hence, we need to add 1 to all end values in
    #    python.
    #
    # Or simply the combination of both: All timing values (like ICs, ...) need to be adjusted by -1.
    # All start values of intervals need to be adjusted by -1.
    # All end values of intervals stay the same, as the two adjustments cancel each other out.
    walking_bouts["start"] -= 1
    ics["ic"] -= 1
    turn_paras["start"] -= 1
    stride_paras["start"] -= 1

    if relative_to_wb is True:
        ics = _relative_to_gs(ics, walking_bouts, "wb_id", columns_to_cut=["ic"])
        turn_paras = _relative_to_gs(turn_paras, walking_bouts, "wb_id", columns_to_cut=["start", "end"])
        stride_paras = _relative_to_gs(stride_paras, walking_bouts, "wb_id", columns_to_cut=["start", "end"])

    return ReferenceData(walking_bouts, ics, turn_paras, stride_paras)._replace(**expect_none)


def _relative_to_gs(
    event_data: Optional[pd.DataFrame],
    gait_sequences: pd.DataFrame,
    gs_index_col: str,
    *,
    columns_to_cut: Sequence[str],
) -> Optional[pd.DataFrame]:
    """Convert the start and end values or event values to values relative to the start of GSs or WBs.

    Note, that this assumes that the input data already has an index level indicating to which GS/WB the entry belongs
    to.
    It does not check if events are actually within the provided GSs/WBs.

    Parameters
    ----------
    event_data
        The event data to convert.
        This can be any dataframe with a multi-index, where at least one level is the GS/WB level
        (specified by ``gs_index_col``).
    gait_sequences
        The gait sequences to use for the conversion.
        The index values must match the index values provided in the ``gs_index_col`` of the event_data.
    gs_index_col
        The name of the index level in the ``event_data`` that contains the GS/WB index.
    columns_to_cut
        The columns to convert.
        This must be a subset of the columns in the ``event_data``.
        For each of the columns we simply subtract the start value of the corresponding GS/WB.
        Make sure that the units of the columns match the units of the GS/WB start values.

    Returns
    -------
    event_data
        A copy of the event data with the converted values.

    """
    if event_data is None:
        return None
    value_to_subtract = gait_sequences["start"].loc[event_data.index.get_level_values(gs_index_col)].to_numpy()
    event_data = event_data.copy()
    event_data[columns_to_cut] = event_data[columns_to_cut].sub(value_to_subtract, axis=0)
    return event_data


@matlab_dataset_docfiller
class BaseGenericMobilisedDataset(BaseGaitDatasetWithReference):
    """Common base class for Datasets based on the Mobilise-D matlab format.

    This should not be used directly, but can be used for custom dataset implementations.

    Parameters
    ----------
    %(file_loader_args)s
    %(dataset_memory_args)s
    %(general_dataset_args)s

    Attributes
    ----------
    %(dataset_data_attrs)s

    """

    raw_data_sensor: Literal["SU", "INDIP", "INDIP2"]
    reference_system: Optional[Literal["INDIP", "Stereophoto"]]
    reference_para_level: Literal["wb", "lwb"]
    sensor_positions: Sequence[str]
    single_sensor_position: str
    sensor_types: Sequence[Literal["acc", "gyr", "mag", "bar"]]
    missing_sensor_error_type: Literal["raise", "warn", "ignore", "skip"]
    missing_reference_error_type: Literal["raise", "warn", "ignore", "skip"]
    memory: joblib.Memory

    _not_expected_per_ref_system: ClassVar[Optional[list[tuple[str, list[str]]]]] = None

    def __init__(
        self,
        *,
        raw_data_sensor: Literal["SU", "INDIP", "INDIP2"] = "SU",
        reference_system: Optional[Literal["INDIP", "Stereophoto"]] = None,
        reference_para_level: Literal["wb", "lwb"] = "wb",
        sensor_positions: Sequence[str] = ("LowerBack",),
        single_sensor_position: str = "LowerBack",
        sensor_types: Sequence[Literal["acc", "gyr", "mag", "bar"]] = ("acc", "gyr"),
        missing_sensor_error_type: Literal["raise", "warn", "ignore", "skip"] = "raise",
        missing_reference_error_type: Literal["raise", "warn", "ignore", "skip"] = "ignore",
        memory: joblib.Memory = joblib.Memory(None),
        groupby_cols: Optional[Union[list[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ) -> None:
        self.raw_data_sensor = raw_data_sensor
        self.reference_system = reference_system
        self.reference_para_level = reference_para_level
        self.sensor_positions = sensor_positions
        self.single_sensor_position = single_sensor_position
        self.sensor_types = sensor_types
        self.memory = memory
        self.missing_sensor_error_type = missing_sensor_error_type
        self.missing_reference_error_type = missing_reference_error_type

        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def _paths_list(self) -> list[Path]:
        raise NotImplementedError

    @property
    def _test_level_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    def _metadata_level_names(self) -> Optional[tuple[str, ...]]:
        raise NotImplementedError

    def _get_file_index_metadata(self, path: Path) -> tuple[str, ...]:
        """Return the metadata for a single file that should be included as index columns.

        This method will be called during index creation for each file returned by `_paths_list`.
        Note, that this will only happen, if `_metadata_level_names` is not None.

        The length of the returned tuple must match the length of `_metadata_level_names`.

        """
        raise NotImplementedError

    @property
    def data(self) -> MobilisedTestData.imu_data:
        return self._load_selected_data("data").imu_data

    @property
    def data_ss(self) -> pd.DataFrame:
        return self.data[self.single_sensor_position]

    def _get_not_expected_for_ref_system(self, ref_system: Optional[str]) -> list[str]:
        if not self._not_expected_per_ref_system:
            return []
        return dict(self._not_expected_per_ref_system).get(ref_system, [])

    @property
    def raw_reference_parameters_(self) -> MobilisedTestData.raw_reference_parameters:
        if self.reference_system is None:
            raise ValueError(
                "The raw_reference_parameters_ and all attributes that depend on it are only available, if a reference "
                "system is specified. "
                "Specify a reference system when creating the dataset object."
            )
        ref_data = self._load_selected_data("reference_parameters_").raw_reference_parameters
        if ref_data is None:
            raise ValueError(
                "Reference data is missing for this test. "
                "If you want to skip this test when iterating over the dataset/in the index, set "
                "`missing_reference_error_type` to `skip`."
            )
        return ref_data

    @property
    def reference_parameters_(self) -> ReferenceData:
        return parse_reference_parameters(
            self.raw_reference_parameters_[self.reference_para_level],
            data_sampling_rate_hz=self.sampling_rate_hz,
            ref_sampling_rate_hz=self.reference_sampling_rate_hz_,
            relative_to_wb=False,
            debug_info=str(self.group_label),
            not_expected_fields=self._get_not_expected_for_ref_system(self.reference_system),
            ignore_expected_warnings=True,
        )

    @property
    def reference_parameters_relative_to_wb_(self) -> ReferenceData:
        return parse_reference_parameters(
            self.raw_reference_parameters_[self.reference_para_level],
            data_sampling_rate_hz=self.sampling_rate_hz,
            ref_sampling_rate_hz=self.reference_sampling_rate_hz_,
            relative_to_wb=True,
            debug_info=str(self.group_label),
            not_expected_fields=self._get_not_expected_for_ref_system(self.reference_system),
            ignore_expected_warnings=True,
        )

    @property
    def sampling_rate_hz(self) -> float:
        return self._load_selected_data("sampling_rate_hz").metadata["sampling_rate_hz"]

    @property
    def reference_sampling_rate_hz_(self) -> float:
        return self._load_selected_data("reference_sampling_rate_hz_").metadata["reference_sampling_rate_hz"]

    def _get_measurement_condition(self) -> str:
        """Return the measurement condition for a single file."""
        raise NotImplementedError

    @property
    def recording_metadata(self) -> MobilisedMetadata:
        self.assert_is_single(None, "recording_metadata")
        recording_identifier = tuple(getattr(self.group_label, s) for s in self._test_level_names)
        return {
            **self._load_selected_data("metadata").metadata,
            "measurement_condition": self._get_measurement_condition(),
            "recording_identifier": recording_identifier,
        }

    @property
    def recording_metadata_as_df(self) -> pd.DataFrame:
        recording_metadata = {p.group_label: pd.Series(p.recording_metadata) for p in self}
        df = pd.concat(recording_metadata, axis=1, names=self.index.columns.to_list()).T
        index_correct_dtype = pd.MultiIndex.from_frame(df.index.to_frame().astype("string"))
        df.index = index_correct_dtype
        return df

    def _get_cohort(self) -> Optional[str]:
        try:
            return self.index_as_tuples()[0].cohort
        except AttributeError:
            warnings.warn(
                "None of the index levels is called `cohort` so we could not extract the relevant metadata. "
                "Cohort is set to `None`. "
                "If you have cohort information for your participants, but there are not part of the index, "
                "subclass the dataset and provide a custom implementation for the `_get_cohort` method.",
                stacklevel=1,
            )
            return None

    @property
    def participant_metadata(self) -> MobilisedParticipantMetadata:
        self.assert_is_single(
            list(self._metadata_level_names) if self._metadata_level_names else self.index.columns.to_list(),
            "participant_metadata",
        )
        # We assume an `infoForAlgo.mat` file is always in the same folder as the data.mat file.
        info_for_algo_file = self.selected_meta_data_file

        participant_metadata = load_mobilised_participant_metadata_file(info_for_algo_file)

        first_level_selected_test_name = self.index.iloc[0][next(iter(self._test_level_names))]

        meta_data = participant_metadata[first_level_selected_test_name]
        final_dict: MobilisedParticipantMetadata = {
            "sensor_height_m": meta_data["SensorHeight"] / 100,
            "height_m": meta_data["Height"] / 100,
            "cohort": self._get_cohort(),
            "handedness": cast(
                "Optional[Literal['left', 'right']]", {"L": "left", "R": "right"}.get(meta_data.get("Handedness"), None)
            ),
            "foot_length_cm": meta_data.get("FootSize"),
            "weight_kg": meta_data.get("Weight"),
            "indip_data_used": meta_data.get("INDIP_DataUsed"),
            "sensor_attachment_su": meta_data.get("SensorAttachment_SU"),
            "sensor_type_su": meta_data.get("SensorType_SU"),
            "walking_aid_used": {0: False, 1: True}.get(int(meta_data.get("WalkingAid_01", -1)), None),
        }

        # Sort dict by key to ensure consistent order
        final_dict = dict(sorted(final_dict.items()))
        return final_dict

    @property
    def participant_metadata_as_df(self) -> pd.DataFrame:
        if self._metadata_level_names:
            names = list(self._metadata_level_names)
            participant_metadata = {p.group_label: pd.Series(p.participant_metadata) for p in self.groupby(names)}
            df = pd.concat(participant_metadata, axis=1, names=names).T
            index_correct_dtype = pd.MultiIndex.from_frame(df.index.to_frame().astype("string"))
            df.index = index_correct_dtype
            return df
        # In this case we assume we just have a single participant
        return pd.Series(self[0].participant_metadata).to_frame()

    def _cached_data_load_no_checks(
        self, path: PathLike
    ) -> tuple[dict[tuple[str, ...], MobilisedTestData], dict[tuple[str, ...], MobilisedAvailableData]]:
        return hybrid_cache(self.memory, 1)(_load_test_data_without_checks)(
            path,
            raw_data_sensor=self.raw_data_sensor,
            reference_system=self.reference_system,
            sensor_positions=self.sensor_positions,
            sensor_types=self.sensor_types,
        )

    def create_precomputed_test_list(self, n_jobs: Optional[int] = None) -> None:
        """Compute and store a json test list for a data.mat file.

        This function should be used by Dataset developers to precompute the test list for a data.mat file.
        This can massively reduce initial loading time, as the dataset index can be generated without loading the data.

        When this is used to generate the test-list, the ``_relpath_to_precomputed_test_list`` method must be
        implemented.

        .. warning:: Don't create test lists for datasets that are likely to be changed.
           Otherwise, you might end up with outdated files and hard to debug errors.
           If you want to recreate the test list (either because of a mobgap or a dataset update), delete all test list
           files and recreate them using this method.

        """
        rel_out_path = self._relpath_to_precomputed_test_list()

        import json

        from joblib import Parallel, delayed

        def process_path(p: str, rel_out_path: str) -> Path:
            _, available_data_per_test = _load_test_data_without_checks(p)
            # Reformat for json dumping:
            available_data_per_test = {
                "data": [
                    {"key": test_name, "value": data._asdict()} for test_name, data in available_data_per_test.items()
                ],
                "__mobgap_version": mobgap.__version__,
            }

            out_path = Path(p).parent / rel_out_path
            with out_path.open("w") as f:
                json.dump(available_data_per_test, f, indent=4)
            return out_path

        pbar = tqdm(
            Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(process_path)(p, rel_out_path) for p in self._paths_list
            ),
            total=len(self._paths_list),
            desc="Creating precomputed test list.",
        )

        for path in pbar:
            pbar.set_postfix_str(f"Processed {path}")

    def _get_precomputed_available_tests(self, path: PathLike) -> dict[tuple[str, ...], MobilisedAvailableData]:
        import json

        test_list_path = Path(path).parent / self._relpath_to_precomputed_test_list()

        with test_list_path.open() as f:
            content = json.load(f)

        # Note: The files contain the mobgap version, that was used to create it. We don't use this yet, but we might
        #       use this in the future, in case the format changes.
        available_data_per_test = content["data"]

        out = {}
        for data in available_data_per_test:
            available_sensors = [tuple(d) for d in data["value"]["available_sensors"]]
            out[tuple(data["key"])] = MobilisedAvailableData(
                available_sensors=available_sensors,
                available_reference_systems=data["value"]["available_reference_systems"],
            )

        return out

    def _relpath_to_precomputed_test_list(self) -> PathLike:
        raise NotImplementedError

    def _get_test_list(self, path: PathLike) -> list[tuple[str, ...]]:
        try:
            available_data = self._get_precomputed_available_tests(path)
        except NotImplementedError:
            available_data = self._cached_data_load_no_checks(path)[1]

        available_keys = []
        for key, value in available_data.items():
            if _check_test_data(
                value,
                raw_data_sensor=self.raw_data_sensor,
                reference_system=self.reference_system,
                sensor_positions=self.sensor_positions,
                missing_sensor_error_type=self.missing_sensor_error_type
                if self.missing_sensor_error_type == "skip"
                else "ignore",
                missing_reference_error_type=self.missing_reference_error_type
                if self.missing_reference_error_type == "skip"
                else "ignore",
            ):
                available_keys.append(key)

        return available_keys

    def _load_selected_data(self, property_name: str) -> MobilisedTestData:
        selected_file = self._get_selected_data_file(property_name)

        selected_test = next(self.index[list(self._test_level_names)].itertuples(index=False, name=None))
        # We use two-level caching here.
        # If the file was loaded before (even in a previous execution) and memory caching is enabled, we get the cached
        # result.
        # If we get the multiple times in the same execution and on the same Dataset object, we use the lru_cache to
        # to keep the current file in memory.
        # The second part is important, when we use the same Dataset object to load multiple parts of the file (e.g.
        # the raw data and the reference parameters).
        data, available = self._cached_data_load_no_checks(selected_file)

        # And then we do the checks afterward.
        # This way, changing the check parameters will not invalidate the cache.
        # The final checks are cheap, so we don't need to cache them.
        test_data = data[selected_test]
        available_data = available[selected_test]

        if not _check_test_data(
            available_data,
            raw_data_sensor=self.raw_data_sensor,
            reference_system=self.reference_system,
            sensor_positions=self.sensor_positions,
            missing_sensor_error_type=self.missing_sensor_error_type,
            missing_reference_error_type=self.missing_reference_error_type,
            error_context=f"Test: {selected_test}, Selected Index: {self.group_label}",
        ):
            # If this returns false, aka test should be skipped, users have manipulated the index.
            raise RuntimeError(
                "A test listed in the index was marked as skipped when loading. "
                "This should not happen and might indicate that the index was manually modified, or "
                "the ``missing_sensor_error_type`` or ``missing_reference_error_type`` was set to "
                "``skip`` AFTER the object was initialized. "
                "This is not supported."
            )

        return test_data

    @property
    def selected_data_file(self) -> Path:
        return self._get_selected_data_file("selected_data_file")

    @property
    def selected_meta_data_file(self) -> Path:
        # We assume an `infoForAlgo.mat` file is always in the same folder as the data.mat file.
        selected_data_file = self._get_selected_data_file("selected_meta_data_file")
        meta_data_file = selected_data_file.parent / "infoForAlgo.mat"

        if not meta_data_file.exists():
            raise FileNotFoundError(
                f"Could not find the participant metadata file {meta_data_file} for the selected data file "
                f"{selected_data_file}. "
                "We assume that this file is always in the same folder as the `data.mat` file."
            )
        return meta_data_file

    def _get_selected_data_file(self, property_name: str) -> Path:
        self.assert_is_single(
            list(self._metadata_level_names) if self._metadata_level_names else self.index.columns.to_list(),
            property_name,
        )
        # We basically make an inverse lookup of the metadata.
        all_path_metadata = self._get_all_path_metadata()

        # Note, we don't check if the metadata is unique here, because we already do that in _get_all_path_metadata.
        if all_path_metadata is None:
            # If no metadata is available, there can only be one file.
            return self._paths_list[0]

        current_selection_metadata = self.index.iloc[0][list(self._metadata_level_names)]

        selected_file = all_path_metadata[(all_path_metadata == current_selection_metadata).all(axis=1)].index[0]

        return selected_file

    def _get_all_path_metadata(self) -> Optional[pd.DataFrame]:
        if self._metadata_level_names is None:
            if len(self._paths_list) > 1:
                raise ValueError(
                    "It seems like no metadata for the files was provided, but there are multiple files. "
                    "We can not distinguish between the files in this case and build a correct index. "
                    "Provide sufficient information that metadata can be loaded for each file. "
                    "How this works depends on the implementation of the Dataset class you are using."
                )
            return None

        metadata_per_level = [
            {
                "__path": path,
                **dict(zip(self._metadata_level_names, self._get_file_index_metadata(path))),
            }
            for path in self._paths_list
        ]
        metadata_per_level = pd.DataFrame.from_records(metadata_per_level).set_index("__path")

        # We test that the meta data for each path is unique. Otherwise we will run into problems later.
        if metadata_per_level.duplicated().any():
            raise ValueError(
                "The metadata for each file path must be unique. "
                "Otherwise, the correct file can not be identified from the selected index rows. "
                "The following paths have duplicate metadata:\n"
                f"{metadata_per_level[metadata_per_level.duplicated(keep=False)]}."
            )

        return metadata_per_level

    def create_index(self) -> pd.DataFrame:
        """Create the dataset index.

        The index columns will consist of the metadata extracted from the columns and the test names.
        """
        # Resolve metadata (aka) test list from loading the files.
        test_name_metadata = (
            pd.concat(
                {
                    path: pd.DataFrame(self._get_test_list(path), columns=list(self._test_level_names))
                    for path in self._paths_list
                }
            )
            .reset_index(level=-1, drop=True)
            .rename_axis(index="__path")
        )

        # Resolve metadata based on the implementation of the child class.
        metadata_per_level = self._get_all_path_metadata()
        if metadata_per_level is None:
            return test_name_metadata.reset_index(drop=True).astype("string")

        return (
            metadata_per_level.merge(test_name_metadata, left_index=True, right_index=True)
            .reset_index(drop=True)
            .astype("string")
        )


@matlab_dataset_docfiller
class GenericMobilisedDataset(BaseGenericMobilisedDataset):
    """A generic dataset loader for the Mobilise-D data format.

    This allows to create a dataset from multiple data files in the Mobilise-D data format stored within nested folder
    structures.
    The index of the dataset will be derived from the folder structure (``parent_folders_as_metadata``) and the
    test names (``test_level_names``).

    All data-attributes are only available, if just a single test is selected.
    Attributes with a trailing underscore (e.g. ``reference_parameters_``) indicate that they contain information from
    external reference systems and not just the IMU.

    Notes
    -----
    The current implementation has two main limitations:

    1. To get the test names, we need to load the entire data file.
       If you have many large data files, this can take a long time.
       To avoid doing that over and over again, we highly recommend to use the ``memory`` parameter to cache the
       results.
       Even with that, the first creation of a dataset, can take a long time, and you need to remember to clean the
       cache, when you change the content of the data files (without changing their paths).
       If you run into cases, where the load time is unreasonable, please open a Github issue, then we can try to
       improve the implementation.
    2. To make sure that the output of the ``index`` property is consistent across multiple machines, you need to make
       sure that the paths are always sorted in the same way.
       Hence, you should always use ``list(sorted(Path.glob(...))`` to get the paths.


    Parameters
    ----------
    paths_list
        A list of paths to the data files.
        These should be the path to the actual data.mat files.
        If you want to use ``participant_metadata``, we expect a ``inforForAlgo.mat`` file within the same folder.
    test_level_names
        The names of the test levels in the data files.
        These will be used as the column names in the index.
        Usually, this will be something like ("TimeMeasure", "Test", "Trial").
        The number of levels can vary between datasets.
        For typically Mobilise-D datasets, check the ``COMMON_TEST_LEVEL_NAMES`` class variable.
    measurement_condition
        Whether the data was recorded under laboratory or free-living conditions.
        At the moment, we only support creating datasets with a single measurement condition.
    parent_folders_as_metadata
        When multiple data files are provided, you need metadata to distinguish them.
        This class implementation expects the names of the parend folder(s) to be used as metadata.
        This parameter expects a list of strings, where each string corresponds to one level of the parent folder
        from left to right (i.e. like you would read the path).
        For example, when ``parent_folders_as_metadata=["cohort", "participant_id"]``, then the path
        ``/parent_folder1/cohort_name/participant_id/data.mat`` would be parsed as
        ``{"cohort": "cohort_name", "participant_id": "participant_id"}``.
        If you want to skip a level, you can pass ``None`` instead of a string.
        For example, when ``parent_folders_as_metadata=["cohort", None, "participant_id"]``, then the path
        ``/parent_folder1/cohort_name/ignored_folder/participant_id/data.mat`` would be parsed as
        ``{"cohort": "cohort_name", "participant_id": "participant_id"}``.
        Note, however, that each file needs a unique combination of metadata.
        If the levels you supply don't result in unique combinations, you will get an error during index creation.
        If you only have a single data file, then you can simply set ``parent_folders_as_metadata=None``.

        .. note:: Ideally one of the metadata levels should be called ``cohort`` otherwise, otherwise the cohort
                  information in ``participant_metadata`` will be set to ``None``.
    %(file_loader_args)s
    %(dataset_memory_args)s
    %(general_dataset_args)s

    Attributes
    ----------
    %(dataset_data_attrs)s
    COMMON_TEST_LEVEL_NAMES
        (ClassVar) A dictionary of common test level names for Mobilise-D datasets.
        These can be passed to the ``test_level_names`` parameter.

    See Also
    --------
    %(dataset_see_also)s

    """

    paths_list: Union[PathLike, Sequence[PathLike]]
    test_level_names: Sequence[str]
    parent_folders_as_metadata: Optional[Sequence[Union[str, None]]]
    measurement_condition: Literal["laboratory", "free_living"]

    COMMON_TEST_LEVEL_NAMES: ClassVar[dict[str, tuple[str, ...]]] = {
        "tvs_lab": ("time_measure", "test", "trial"),
        "tvs_2.5h": ("time_measure", "recording"),
    }

    def __init__(
        self,
        paths_list: Union[PathLike, Sequence[PathLike]],
        test_level_names: Sequence[str],
        parent_folders_as_metadata: Optional[Sequence[Union[str, None]]] = None,
        *,
        measurement_condition: Literal["laboratory", "free_living"],
        raw_data_sensor: Literal["SU", "INDIP", "INDIP2"] = "SU",
        reference_system: Optional[Literal["INDIP", "Stereophoto"]] = None,
        reference_para_level: Literal["wb", "lwb"] = "wb",
        sensor_positions: Sequence[str] = ("LowerBack",),
        single_sensor_position: str = "LowerBack",
        sensor_types: Sequence[Literal["acc", "gyr", "mag", "bar"]] = ("acc", "gyr"),
        missing_sensor_error_type: Literal["raise", "warn", "ignore", "skip"] = "raise",
        missing_reference_error_type: Literal["raise", "warn", "ignore", "skip"] = "ignore",
        memory: joblib.Memory = joblib.Memory(None),
        groupby_cols: Optional[Union[list[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ) -> None:
        self.paths_list = paths_list
        self.test_level_names = test_level_names
        self.parent_folders_as_metadata = parent_folders_as_metadata
        self.measurement_condition = measurement_condition
        super().__init__(
            raw_data_sensor=raw_data_sensor,
            reference_system=reference_system,
            reference_para_level=reference_para_level,
            sensor_positions=sensor_positions,
            sensor_types=sensor_types,
            single_sensor_position=single_sensor_position,
            memory=memory,
            groupby_cols=groupby_cols,
            subset_index=subset_index,
            missing_sensor_error_type=missing_sensor_error_type,
            missing_reference_error_type=missing_reference_error_type,
        )

    @property
    def _paths_list(self) -> list[Path]:
        paths_list = self.paths_list

        if isinstance(paths_list, (str, Path)):
            paths_list = [paths_list]
        elif not isinstance(paths_list, Sequence):
            raise TypeError(
                f"paths_list must be a PathLike or a Sequence of PathLikes, but got {type(paths_list)}. "
                "For the list of paths, you need to make sure that it is persistent and can be iterated over "
                "multiple times. "
                "So don't use a generator or directly pass the output of `Path.glob`. "
                "Instead use `list(sorted(Path.glob(...)))` to get the paths."
            )

        return [Path(path) for path in paths_list]

    @property
    def _test_level_names(self) -> tuple[str, ...]:
        return tuple(self.test_level_names)

    def _get_measurement_condition(self) -> str:
        return self.measurement_condition

    @property
    def _metadata_level_names(self) -> Optional[tuple[str, ...]]:
        if self.parent_folders_as_metadata is None:
            return None
        return tuple(name for name in self.parent_folders_as_metadata if name is not None)

    def _get_file_index_metadata(self, path: Path) -> tuple[str, ...]:
        """Select the metadata from the file path.

        We pick all the parent folder names for the entries in `self.parent_folders_as_metadata` for which the value
        is not None.

        """
        start_level = len(self.parent_folders_as_metadata) - 1
        return tuple(
            path.parents[start_level - level].name
            for level, level_name in enumerate(self.parent_folders_as_metadata)
            if level_name is not None
        )
