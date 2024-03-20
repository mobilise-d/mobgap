import warnings
from collections.abc import Iterator, Sequence
from functools import lru_cache, partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Literal,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
)

import joblib
import numpy as np
import pandas as pd
import scipy.io as sio

from mobgap._docutils import make_filldoc
from mobgap.data.base import BaseGaitDatasetWithReference, ReferenceData, base_gait_dataset_docfiller_dict

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
    missing_sensor_error_type
        Whether to throw an error ("raise"), a warning ("warn") or ignore ("ignore") when a sensor is missing.

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
    metadata: MobilisedMetadata


def load_mobilised_participant_metadata_file(path: PathLike) -> dict[str, dict[str, Any]]:
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


@matlab_dataset_docfiller
def load_mobilised_matlab_format(
    path: PathLike,
    *,
    raw_data_sensor: Optional[Literal["SU", "INDIP", "INDIP2"]] = "SU",
    reference_system: Optional[Literal["INDIP", "Stereophoto"]] = None,
    sensor_positions: Sequence[str] = ("LowerBack",),
    sensor_types: Sequence[Literal["acc", "gyr", "mag", "bar"]] = ("acc", "gyr"),
    missing_sensor_error_type: Literal["raise", "warn", "ignore"] = "raise",
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
    if raw_data_sensor is None and reference_system is None:
        raise ValueError(
            "At least one of raw_data_sensor and reference_system must be set. Otherwise no data is loaded."
        )

    if raw_data_sensor:
        raw_data_sensor = ("SU_" + raw_data_sensor) if raw_data_sensor != "SU" else "SU"

    sensor_test_level_marker = raw_data_sensor or "SU"

    data = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False, mat_dtype=True)
    data_per_test = _parse_until_test_level(data["data"], (sensor_test_level_marker, "Standards"))

    data_per_test_dict = {
        test_name: _process_test_data(
            test_data,
            test_name,
            raw_data_sensor=raw_data_sensor,
            reference_system=reference_system,
            sensor_positions=sensor_positions,
            sensor_types=sensor_types,
            missing_sensor_error_type=missing_sensor_error_type,
        )
        for test_name, test_data in data_per_test
    }

    # Unit conversion of imu data, if available
    for test, data in data_per_test_dict.items():
        if data.imu_data is not None:
            for sensor_position in data.imu_data:
                # Convert acc columns to m/s-2
                data_per_test_dict[test].imu_data[sensor_position][["acc_x", "acc_y", "acc_z"]] *= 9.81

    return data_per_test_dict


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


def _process_test_data(  # noqa: C901, PLR0912, PLR0915
    test_data: sio.matlab.mat_struct,
    test_name: tuple[str, ...],
    *,
    raw_data_sensor: Optional[str],
    reference_system: Optional[str],
    sensor_positions: Sequence[str],
    sensor_types: Sequence[Literal["acc", "gyr", "mag", "bar"]],
    missing_sensor_error_type: Literal["raise", "warn", "ignore"] = "raise",
) -> MobilisedTestData:
    if missing_sensor_error_type not in ["raise", "warn", "ignore"]:
        raise ValueError(f"Invalid value for missing_sensor_error_type: {missing_sensor_error_type}")

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
                error_message = f"Sensor position {sensor_pos} is not available for test {test_name}."

                if missing_sensor_error_type == "raise":
                    raise ValueError(error_message) from e
                if missing_sensor_error_type == "warn":
                    warnings.warn(error_message, stacklevel=1)

            else:
                all_imu_data[sensor_pos] = _parse_single_sensor_data(raw_data, sensor_types)
                sampling_rates_obj = raw_data.Fs
                sampling_rates.update(
                    {f"{sensor_pos}_{k}": getattr(sampling_rates_obj, k) for k in sampling_rates_obj._fieldnames}
                )

        if all_imu_data == {}:
            error_message = (
                f"Expected at least one valid sensor position for {raw_data_sensor}. Given: {sensor_positions}"
            )
            if missing_sensor_error_type == "raise":
                raise ValueError(error_message)
            if missing_sensor_error_type == "warn":
                warnings.warn(error_message, stacklevel=1)

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

    return MobilisedTestData(
        imu_data=all_imu_data, raw_reference_parameters=reference_data, metadata=MobilisedMetadata(**meta_data)
    )


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
    return [_parse_matlab_struct(wb_data) for wb_data in reference_data]


def _parse_matlab_struct(struct: sio.matlab.mat_struct) -> dict[str, Any]:
    """Parse a simple matlab struct that only contains simple types (no nested structs or arrays)."""
    return {k: getattr(struct, k) for k in struct._fieldnames}


@lru_cache(maxsize=1)
def cached_load_current(selected_file: PathLike, loader_function: Callable[[PathLike], T]) -> T:
    # TODO: Check if we actually get a proper cache hit here and it really helps performance.
    return loader_function(selected_file)


def _ensure_is_list(value: Any) -> list:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list):
        return [value]
    return value


def parse_reference_parameters(
    ref_data: list[dict[str, Union[str, float, int, np.ndarray]]],
    *,
    ref_sampling_rate_hz: float,
    data_sampling_rate_hz: float,
    relative_to_wb: bool = False,
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

    for wb_id, wb in enumerate(ref_data, start=1):
        walking_bouts.append(
            {
                "wb_id": wb_id,
                "start": wb["Start"],
                "end": wb["End"],
                "n_strides": int(wb["NumberStrides"]),
                "duration_s": wb["Duration"],
                "length_m": wb["Length"],
                "avg_speed_mps": wb["WalkingSpeed"],
                "avg_cadence_spm": wb["Cadence"],
                "avg_stride_length_m": wb["AverageStrideLength"],
                "termination_reason": wb["TerminationReason"],
            }
        )

        ic_vals = _ensure_is_list(wb["InitialContact_Event"])
        ics.append(
            pd.DataFrame.from_dict(
                {
                    "wb_id": [wb_id] * len(ic_vals),
                    "ic": ic_vals,
                    "lr_label": _ensure_is_list(wb["InitialContact_LeftRight"]),
                }
            )
        )
        turn_starts = _ensure_is_list(wb["Turn_Start"])
        turn_paras.append(
            pd.DataFrame.from_dict(
                {
                    "wb_id": [wb_id] * len(turn_starts),
                    "start": turn_starts,
                    "end": _ensure_is_list(wb["Turn_End"]),
                    "duration_s": _ensure_is_list(wb["Turn_Duration"]),
                    "angle_deg": _ensure_is_list(wb["Turn_Angle"]),
                }
            )
        )
        starts, ends = zip(*_ensure_is_list(wb["Stride_InitialContacts"]))
        stride_paras.append(
            pd.DataFrame.from_dict(
                {
                    "wb_id": [wb_id] * len(starts),
                    "start": starts,
                    "end": ends,
                    "duration_s": _ensure_is_list(wb["Stride_Duration"]),
                    "length_m": _ensure_is_list(wb["Stride_Length"]),
                    "speed_mps": _ensure_is_list(wb["Stride_Speed"]),
                    "stance_time_s": _ensure_is_list(wb["Stance_Duration"]),
                    "swing_time_s": _ensure_is_list(wb["Swing_Duration"]),
                }
            )
        )

    walking_bouts = pd.DataFrame.from_records(walking_bouts).set_index("wb_id")
    walking_bouts[["start", "end"]] = (walking_bouts[["start", "end"]] * data_sampling_rate_hz).round().astype(int)

    ics = pd.concat(ics, ignore_index=True)
    ics_is_na = ics["ic"].isna()
    ics = ics[~ics_is_na].drop_duplicates()
    ics["ic"] = (ics["ic"] * data_sampling_rate_hz).round().astype(int)
    ics.index.name = "step_id"
    ics = ics.reset_index().set_index(["wb_id", "step_id"])
    # make left-right labels lowercase
    ics["lr_label"] = ics["lr_label"].str.lower()

    turn_paras = pd.concat(turn_paras, ignore_index=True)
    turn_paras.index.name = "turn_id"
    turn_paras = turn_paras.reset_index().set_index(["wb_id", "turn_id"])

    stride_paras = pd.concat(stride_paras, ignore_index=True)
    stride_ics_is_na = stride_paras[["start", "end"]].isna().any(axis=1)
    stride_paras = stride_paras[~stride_ics_is_na]
    # Note: For the INDIP system it seems like start and end are provided in samples already and not in seconds.
    #       I am not sure what the correct behavior is, but we try to handle it to avoid confusion on the user side.
    #       Unfortunately, there is no 100% reliable way to detect this, so we just check if the values are in the IC
    #       list (which seems to be provided in time in all ref systems I have seen).
    #
    # If we assume the values are in samples of the reference system, than we attempt to convert them to samples of the
    # single sensor system.
    # For the INDIP system, that shouldn't matter, as the sampling rates are the same, but hey you can never be too
    # safe.
    assume_stride_paras_in_samples = (
        (stride_paras["start"] * (data_sampling_rate_hz / ref_sampling_rate_hz)).round().astype(int)
    )
    # ICs are already converted to samples here -> I.e. if they are not all in here, we assume that the stride
    # parameters are also in seconds not in samples.
    if not assume_stride_paras_in_samples.isin(ics["ic"]).all():
        stride_paras[["start", "end"]] = (stride_paras[["start", "end"]] * data_sampling_rate_hz).round().astype(int)
        # We check again, just to be sure and if they are still not there, we throw an error.
        if not stride_paras["start"].isin(ics["ic"]).all():
            raise ValueError(
                "There seems to be a mismatch between the provided stride parameters and the provided initial "
                "contacts of the reference system. "
                "At least some of the ICs marking the start of a stride are not found in the dedicated IC list."
            )
    else:
        stride_paras[["start", "end"]] = (
            (stride_paras[["start", "end"]] * (data_sampling_rate_hz / ref_sampling_rate_hz)).round().astype(int)
        )

    # We also get the correct LR-label for the stride parameters from the ICs.
    ic_duplicate_as_nan = ics.copy()
    # We set the values to Nan first and then drop one of the duplicates.
    ic_duplicate_as_nan.loc[ics["ic"].duplicated(keep=False), "lr_label"] = pd.NA
    ic_duplicate_as_nan = ic_duplicate_as_nan.drop_duplicates()
    if ic_duplicate_as_nan["lr_label"].isna().any():
        warnings.warn(
            "There were multiple ICs with the same index value, but different LR labels. "
            "This is likely an issue with the reference system you should further investigate. "
            "For now, we set the `lr_label` of the stride corresponding to this IC to Nan. "
            "However, both values still remain in the IC list.",
            stacklevel=2,
        )
    stride_paras["lr_label"] = ic_duplicate_as_nan.set_index("ic").loc[stride_paras["start"], "lr_label"].to_numpy()
    stride_paras.index.name = "s_id"
    stride_paras = stride_paras.reset_index().set_index(["wb_id", "s_id"])

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

    return ReferenceData(walking_bouts, ics, turn_paras, stride_paras)


def _relative_to_gs(
    event_data: pd.DataFrame, gait_sequences: pd.DataFrame, gs_index_col: str, *, columns_to_cut: Sequence[str]
) -> pd.DataFrame:
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
    sensor_types: Sequence[Literal["acc", "gyr", "mag", "bar"]]
    memory: joblib.Memory

    def __init__(
        self,
        *,
        raw_data_sensor: Literal["SU", "INDIP", "INDIP2"] = "SU",
        reference_system: Optional[Literal["INDIP", "Stereophoto"]] = None,
        reference_para_level: Literal["wb", "lwb"] = "wb",
        sensor_positions: Sequence[str] = ("LowerBack",),
        sensor_types: Sequence[Literal["acc", "gyr", "mag", "bar"]] = ("acc", "gyr"),
        missing_sensor_error_type: Literal["raise", "warn", "ignore"] = "raise",
        memory: joblib.Memory = joblib.Memory(None),
        groupby_cols: Optional[Union[list[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ) -> None:
        self.raw_data_sensor = raw_data_sensor
        self.reference_system = reference_system
        self.reference_para_level = reference_para_level
        self.sensor_positions = sensor_positions
        self.sensor_types = sensor_types
        self.memory = memory
        self.missing_sensor_error_type = missing_sensor_error_type

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
    def raw_reference_parameters_(self) -> MobilisedTestData.raw_reference_parameters:
        if self.reference_system is None:
            raise ValueError(
                "The raw_reference_parameters_ and all attributes that depend on it are only available, if a reference "
                "system is specified. "
                "Specify a reference system when creating the dataset object."
            )
        return self._load_selected_data("reference_parameters_").raw_reference_parameters

    @property
    def reference_parameters_(self) -> ReferenceData:
        return parse_reference_parameters(
            self.raw_reference_parameters_[self.reference_para_level],
            data_sampling_rate_hz=self.sampling_rate_hz,
            ref_sampling_rate_hz=self.reference_sampling_rate_hz_,
            relative_to_wb=False,
        )

    @property
    def reference_parameters_relative_to_wb_(self) -> ReferenceData:
        return parse_reference_parameters(
            self.raw_reference_parameters_[self.reference_para_level],
            data_sampling_rate_hz=self.sampling_rate_hz,
            ref_sampling_rate_hz=self.reference_sampling_rate_hz_,
            relative_to_wb=True,
        )

    @property
    def sampling_rate_hz(self) -> float:
        return self._load_selected_data("sampling_rate_hz").metadata.sampling_rate_hz

    @property
    def reference_sampling_rate_hz_(self) -> float:
        return self._load_selected_data("reference_sampling_rate_hz_").metadata.reference_sampling_rate_hz

    @property
    def metadata(self) -> MobilisedMetadata:
        return self._load_selected_data("metadata").metadata

    @property
    def participant_metadata(self) -> dict[str, Any]:
        # We assume an `infoForAlgo.mat` file is always in the same folder as the data.mat file.
        info_for_algo_file = self.selected_meta_data_file

        participant_metadata = load_mobilised_participant_metadata_file(info_for_algo_file)

        first_level_selected_test_name = self.index.iloc[0][next(iter(self._test_level_names))]

        return participant_metadata[first_level_selected_test_name]

    @property
    def _cached_data_load(self) -> Callable[[PathLike], dict[tuple[str, ...], MobilisedTestData]]:
        return partial(
            self.memory.cache(load_mobilised_matlab_format),
            raw_data_sensor=self.raw_data_sensor,
            reference_system=self.reference_system,
            sensor_positions=self.sensor_positions,
            sensor_types=self.sensor_types,
            missing_sensor_error_type=self.missing_sensor_error_type,
        )

    def _get_test_list(self, path: PathLike) -> list[tuple[str, ...]]:
        return list(self._cached_data_load(path).keys())

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
        return cached_load_current(selected_file, self._cached_data_load)[selected_test]

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
        self.assert_is_single(None, property_name)
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
            {"__path": path, **dict(zip(self._metadata_level_names, self._get_file_index_metadata(path)))}
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
            return test_name_metadata.reset_index(drop=True)

        return metadata_per_level.merge(test_name_metadata, left_index=True, right_index=True).reset_index(drop=True)


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
        raw_data_sensor: Literal["SU", "INDIP", "INDIP2"] = "SU",
        reference_system: Optional[Literal["INDIP", "Stereophoto"]] = None,
        reference_para_level: Literal["wb", "lwb"] = "wb",
        sensor_positions: Sequence[str] = ("LowerBack",),
        sensor_types: Sequence[Literal["acc", "gyr", "mag", "bar"]] = ("acc", "gyr"),
        missing_sensor_error_type: Literal["raise", "warn", "ignore"] = "raise",
        memory: joblib.Memory = joblib.Memory(None),
        groupby_cols: Optional[Union[list[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ) -> None:
        self.paths_list = paths_list
        self.test_level_names = test_level_names
        self.parent_folders_as_metadata = parent_folders_as_metadata
        super().__init__(
            raw_data_sensor=raw_data_sensor,
            reference_system=reference_system,
            reference_para_level=reference_para_level,
            sensor_positions=sensor_positions,
            sensor_types=sensor_types,
            memory=memory,
            groupby_cols=groupby_cols,
            subset_index=subset_index,
            missing_sensor_error_type=missing_sensor_error_type,
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
