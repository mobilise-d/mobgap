import warnings
from collections.abc import Iterator, Sequence
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal, NamedTuple, Optional, TypeVar, Union

import joblib
import numpy as np
import pandas as pd
import scipy.io as sio
from tpcp import Dataset

from gaitlink._docutils import make_filldoc

T = TypeVar("T")

PathLike = TypeVar("PathLike", str, Path)

docfiller = make_filldoc(
    {
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
    """,
        "general_dataset_args": """
    groupby_cols
        Columns to group the data by. See :class:`~tpcp.Dataset` for details.
    subset_index
        The selected subset of the index. See :class:`~tpcp.Dataset` for details.
    """,
        "dataset_memory_args": """
    memory
        A joblib memory object to cache the results of the data loading.
        This is highly recommended, if you have many large data files.
        Otherwise, the initial index creation can take a long time.
    """,
        "dataset_data_attrs": """
    data
        The raw IMU data.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz.
    reference_parameters_
        The reference parameters (if available).
    reference_sampling_rate_hz_
        The sampling rate of the reference data in Hz.
    metadata
        The metadata of the selected test.
    participant_metadata
        The participant metadata loaded from the `infoForAlgo.mat` file.
    """,
        "dataset_see_also": """
    :class:`~tpcp.Dataset`
        For details about the ``groupby_cols`` and ``subset_index`` parameters.
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
    imu_data: Optional[dict[str, pd.DataFrame]]
    # TODO: Update Any typing once I understand the data better.
    reference_parameters: Optional[dict[Literal["wb", "lwb"], Any]]
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


@docfiller
def load_mobilised_matlab_format(
    path: PathLike,
    *,
    raw_data_sensor: Optional[Literal["SU", "INDIP", "INDIP2"]] = "SU",
    reference_system: Optional[Literal["INDIP", "Stereophoto"]] = None,
    sensor_positions: Sequence[str] = ("LowerBack",),
    sensor_types: Sequence[Literal["acc", "gyr", "mag", "bar"]] = ("acc", "gyr"),
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
                # Get all available sensor positions and pick the one containing the sensor position string. 
                # If multiple sensor positions matching the string are available, raise an error.
                sensor_positions_list = all_sensor_data._fieldnames
                position_in = [pos for pos in sensor_positions_list if sensor_pos in pos]
                if len(position_in) > 1:
                    raise ValueError(
                        f"""Sensor position '{sensor_pos}' similar to multiple positions ({position_in})
                        for test {test_name}."""
                    )

                raw_data = getattr(all_sensor_data, position_in[0])
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

    return MobilisedTestData(
        imu_data=all_imu_data, reference_parameters=reference_data, metadata=MobilisedMetadata(**meta_data)
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
    # We convert acc data to m/s^2
    if "acc" in sensor_types:
        parsed_data[["acc_x", "acc_y", "acc_z"]] *= 9.81

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


@docfiller
class _GenericMobilisedDataset(Dataset):
    """Common base class for Datasets based on the Mobilise-D matlab format.

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
    sensor_positions: Sequence[str]
    sensor_types: Sequence[Literal["acc", "gyr", "mag", "bar"]]
    memory: joblib.Memory

    def __init__(
        self,
        *,
        raw_data_sensor: Literal["SU", "INDIP", "INDIP2"] = "SU",
        reference_system: Optional[Literal["INDIP", "Stereophoto"]] = None,
        sensor_positions: Sequence[str] = ("LowerBack",),
        sensor_types: Sequence[Literal["acc", "gyr", "mag", "bar"]] = ("acc", "gyr"),
        memory: joblib.Memory = joblib.Memory(None),
        groupby_cols: Optional[Union[list[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ) -> None:
        self.raw_data_sensor = raw_data_sensor
        self.reference_system = reference_system
        self.sensor_positions = sensor_positions
        self.sensor_types = sensor_types
        self.memory = memory

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
    def reference_parameters_(self) -> MobilisedTestData.reference_parameters:
        return self._load_selected_data("reference_parameters_").reference_parameters

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


@docfiller
class GenericMobilisedDataset(_GenericMobilisedDataset):
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
    COMMON_TEST_LEVEL_NAMES
        (ClassVar) A dictionary of common test level names for Mobilise-D datasets.
        These can be passed to the ``test_level_names`` parameter.
    %(dataset_data_attrs)s

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
        sensor_positions: Sequence[str] = ("LowerBack",),
        sensor_types: Sequence[Literal["acc", "gyr", "mag", "bar"]] = ("acc", "gyr"),
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
            sensor_positions=sensor_positions,
            sensor_types=sensor_types,
            memory=memory,
            groupby_cols=groupby_cols,
            subset_index=subset_index,
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
