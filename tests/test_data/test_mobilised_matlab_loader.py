import warnings

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from mobgap import PROJECT_ROOT
from mobgap.data import (
    GenericMobilisedDataset,
    get_all_lab_example_data_paths,
    load_mobilised_matlab_format,
    load_mobilised_participant_metadata_file,
    parse_reference_parameters,
)


@pytest.fixture
def example_data_path():
    return get_all_lab_example_data_paths()[("HA", "001")]


@pytest.fixture
def example_missing_data_path():
    potential_paths = (PROJECT_ROOT / "tests" / "test_data" / "data" / "lab_missing_sensor").rglob("data.mat")
    return {(path.parents[1].name, path.parents[0].name): path.parent for path in potential_paths}[("HA", "001")]


def test_simple_file_loading(example_data_path, recwarn, snapshot):
    data = load_mobilised_matlab_format(example_data_path / "data.mat")

    # We don't expect any user-warnings to be raised
    assert len([w for w in recwarn if issubclass(w.category, UserWarning)]) == 0

    assert len(data) == 3

    for name, test_data in data.items():
        assert isinstance(name, tuple)
        assert len(name) == 3

        # Test basic metadata
        assert test_data.metadata["time_zone"] == "Europe/Berlin"

        assert test_data.metadata["sampling_rate_hz"] == 100
        assert isinstance(test_data.metadata["start_date_time_iso"], str)
        # First sample should be equivalent to the start date time
        assert (
            test_data.imu_data["LowerBack"]
            .index.tz_convert(test_data.metadata["time_zone"])
            .round("ms")[0]
            .isoformat(timespec="milliseconds")
            == test_data.metadata["start_date_time_iso"]
        )

        # Test IMU data
        assert list(test_data.imu_data["LowerBack"].columns) == ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
        assert set(test_data.imu_data.keys()) == {"LowerBack"}
        assert len(test_data.imu_data["LowerBack"]) > 100

        snapshot_data = test_data.imu_data["LowerBack"].head(5)
        snapshot_data.index = snapshot_data.index.round("ms")
        snapshot.assert_match(snapshot_data, name)

        # By default, there should be no reference parameter
        assert test_data.raw_reference_parameters is None
        assert test_data.metadata["reference_sampling_rate_hz"] is None


def test_error_if_nothing_to_load(example_data_path):
    with pytest.raises(ValueError, match="At least one of raw_data_sensor and reference_system must be set."):
        _ = load_mobilised_matlab_format(example_data_path / "data.mat", raw_data_sensor=None, reference_system=None)


def test_load_only_reference(example_data_path, recwarn):
    data = load_mobilised_matlab_format(example_data_path / "data.mat", raw_data_sensor=None, reference_system="INDIP")

    # We don't expect any user-warnings to be raised
    assert len([w for w in recwarn if issubclass(w.category, UserWarning)]) == 0

    assert len(data) == 3

    for _name, test_data in data.items():
        assert test_data.imu_data is None
        assert test_data.metadata["sampling_rate_hz"] is None

        assert test_data.raw_reference_parameters is not None
        assert test_data.metadata["reference_sampling_rate_hz"] == 100


def test_reference_system_loading(example_data_path):
    data = load_mobilised_matlab_format(
        example_data_path / "data.mat",
        reference_system="INDIP",
    )

    number_of_tests_with_reference = 0

    for _name, test_data in data.items():
        if test_data.raw_reference_parameters is not None:
            number_of_tests_with_reference += 1

            assert test_data.metadata["reference_sampling_rate_hz"] == 100
            assert set(test_data.raw_reference_parameters.keys()) == {"lwb", "wb"}
            for _key, value in test_data.raw_reference_parameters.items():
                assert isinstance(value, list)
                assert len(value) > 0
                for wb in value:
                    assert isinstance(wb, dict)

    assert number_of_tests_with_reference == 3


def test_load_participant_metadata(example_data_path):
    participant_metadata = load_mobilised_participant_metadata_file(example_data_path / "infoForAlgo.mat")

    assert len(participant_metadata) == 1

    t1_metadata = participant_metadata["TimeMeasure1"]

    # We test the values for a couple of different datatypes to make sure they were parsed correctly
    assert t1_metadata["Height"] == 159.0
    assert t1_metadata["Handedness"] == "R"
    assert t1_metadata["SensorType_SU"] == "MM+"


@pytest.mark.parametrize("ref_system", ["INDIP", "Stereophoto"])
@pytest.mark.parametrize("reference_para_level", ["wb", "lwb"])
@pytest.mark.parametrize("relative_to_wb", [True, False])
def test_parse_reference_data_has_correct_output(ref_system, example_data_path, reference_para_level, relative_to_wb):
    data = load_mobilised_matlab_format(example_data_path / "data.mat", reference_system=ref_system)

    for v in data.values():
        raw_ref_paras = v.raw_reference_parameters[reference_para_level]
        parsed_data = parse_reference_parameters(
            raw_ref_paras,
            data_sampling_rate_hz=100,
            ref_sampling_rate_hz=100,
            relative_to_wb=relative_to_wb,
            debug_info="",
        )

        # All outputs are dfs:
        assert isinstance(parsed_data.turn_parameters, pd.DataFrame)
        assert isinstance(parsed_data.wb_list, pd.DataFrame)
        assert isinstance(parsed_data.ic_list, pd.DataFrame)
        assert isinstance(parsed_data.stride_parameters, pd.DataFrame)
        # For all paraemters, the first index name should be "wb_id"
        assert parsed_data.turn_parameters.index.names[0] == "wb_id"
        assert parsed_data.wb_list.index.names[0] == "wb_id"
        assert parsed_data.ic_list.index.names[0] == "wb_id"
        assert parsed_data.stride_parameters.index.names[0] == "wb_id"

        assert len(parsed_data.wb_list) == len(raw_ref_paras)
        assert len(parsed_data.ic_list) == len(
            pd.concat([pd.Series(wb["InitialContact_Event"]) for wb in raw_ref_paras]).dropna()
        )

        if relative_to_wb is True:
            assert parsed_data.ic_list.iloc[0]["ic"] == 0
            assert parsed_data.stride_parameters.iloc[0]["start"] == 0
        else:
            assert parsed_data.ic_list.iloc[0]["ic"] != 0
            assert parsed_data.stride_parameters.iloc[0]["start"] != 0


def test_parse_reference_paras_uses_correct_sampling_rate(example_data_path):
    data = load_mobilised_matlab_format(example_data_path / "data.mat", reference_system="INDIP")

    raw_ref_paras = next(iter(data.values())).raw_reference_parameters["wb"]

    # With 100 Hz
    parsed_data_100 = parse_reference_parameters(
        raw_ref_paras, data_sampling_rate_hz=100, ref_sampling_rate_hz=100, relative_to_wb=True, debug_info="100Hz"
    )
    # With 50 Hz
    parsed_data_50 = parse_reference_parameters(
        raw_ref_paras, data_sampling_rate_hz=50, ref_sampling_rate_hz=100, relative_to_wb=True, debug_info="50Hz"
    )

    # We cannot test for direct equivalence, because of rounding within the methods.
    # But we can test that there is rougly a factor of two between the two outputs
    assert (parsed_data_50.ic_list["ic"] - np.ceil(parsed_data_100.ic_list["ic"] / 2).astype("int64") <= 1).all()
    assert (
        parsed_data_50.turn_parameters["start"] - np.ceil(parsed_data_100.turn_parameters["start"] / 2).astype("int64")
        <= 1
    ).all()
    assert (
        parsed_data_50.stride_parameters["start"]
        - np.ceil(parsed_data_100.stride_parameters["start"] / 2).astype("int64")
        <= 1
    ).all()
    assert (parsed_data_50.wb_list["start"] - np.ceil(parsed_data_100.wb_list["start"] / 2).astype("int64") <= 1).all()


class TestDatasetClass:
    def test_index_creation(self, example_data_path):
        ds = GenericMobilisedDataset(
            sorted([p / "data.mat" for p in get_all_lab_example_data_paths().values()]),
            GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_lab"],
            ("cohort", "participant_id"),
            measurement_condition="laboratory",
        )

        manually_loaded_participant_index = list(get_all_lab_example_data_paths().keys())

        assert len(ds) == 3 * 3
        assert set(ds.index[["cohort", "participant_id"]].to_records(index=False).tolist()) == set(
            manually_loaded_participant_index
        )

        assert ds.index.columns.tolist() == ["cohort", "participant_id", "time_measure", "test", "trial"]

    def test_participant_metadata(self, example_data_path, snapshot):
        ds = GenericMobilisedDataset(
            sorted([p / "data.mat" for p in get_all_lab_example_data_paths().values()]),
            GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_lab"],
            ("cohort", "participant_id"),
            measurement_condition="bla",
            reference_system="INDIP",
        )

        meta_data = ds[0].participant_metadata
        recording_meta_data = ds[0].recording_metadata

        assert meta_data["cohort"] == ds[0].group_label.cohort
        assert recording_meta_data["measurement_condition"] == ds.measurement_condition

        snapshot.assert_match(pd.Series(meta_data, name="metadata").to_frame())
        snapshot.assert_match(pd.Series(recording_meta_data, name="recording_metadata").to_frame())

    def test_participant_metdata_as_df(self, example_data_path):
        ds = GenericMobilisedDataset(
            sorted([p / "data.mat" for p in get_all_lab_example_data_paths().values()]),
            GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_lab"],
            ("cohort", "participant_id"),
            measurement_condition="bla",
            reference_system="INDIP",
        )

        meta_data = ds.participant_metadata_as_df
        meta_data_first = ds[0].participant_metadata

        recording_metadata = ds.recording_metadata_as_df
        recording_metadata_first = ds[0].recording_metadata

        assert len(meta_data) == len(ds.index[["cohort", "participant_id"]].drop_duplicates())
        assert meta_data.index.names == ["cohort", "participant_id"]
        assert meta_data.iloc[0].to_dict() == meta_data_first

        assert len(recording_metadata) == len(ds)
        assert all(recording_metadata.index.names == ds.index.columns)
        assert recording_metadata.iloc[0].to_dict() == recording_metadata_first

    def test_participant_metadata_warning(self, example_data_path):
        ds = GenericMobilisedDataset(
            sorted([p / "data.mat" for p in get_all_lab_example_data_paths().values()]),
            GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_lab"],
            ("NOT_COHORT", "participant_id"),
            measurement_condition="bla",
            reference_system="INDIP",
        )

        with pytest.warns(UserWarning, match="None of the index levels is called `cohort`."):
            meta_data = ds[0].participant_metadata

        assert meta_data["cohort"] == None

    @pytest.mark.parametrize("reference_para_level", ["wb", "lwb"])
    def test_loaded_data(self, example_data_path, reference_para_level):
        ds = GenericMobilisedDataset(
            example_data_path / "data.mat",
            GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_lab"],
            None,
            measurement_condition="laboratory",
            reference_system="INDIP",
            reference_para_level=reference_para_level,
        )

        manually_data = load_mobilised_matlab_format(example_data_path / "data.mat", reference_system="INDIP")

        assert list(ds.index.itertuples(index=False)) == list(manually_data.keys())

        for test_ds, test in zip(ds, manually_data.values()):
            test_ds: GenericMobilisedDataset
            # Test.metadata should be a subset of recording_metadata
            assert {k: v for k, v in test_ds.recording_metadata.items() if k in test.metadata} == test.metadata
            assert test_ds.sampling_rate_hz == test.metadata["sampling_rate_hz"]
            assert test_ds.reference_sampling_rate_hz_ == test.metadata["reference_sampling_rate_hz"]
            assert test_ds.data.keys() == test.imu_data.keys()
            for sensor in test_ds.data:
                assert_frame_equal(test_ds.data[sensor], test.imu_data[sensor])
            # It is a little hard to compare the entire reference parameters, as they are a list of dicts
            # We just compare the length and the first entry
            for wb_type in test_ds.raw_reference_parameters_:
                assert len(test_ds.raw_reference_parameters_[wb_type]) == len(test.raw_reference_parameters[wb_type])
                assert (
                    test_ds.raw_reference_parameters_[wb_type][0]["Start"]
                    == test.raw_reference_parameters[wb_type][0]["Start"]
                )

            # For the parsed reference parameter, we compare that some basic values are the same in the raw and parsed
            ref_paras = test_ds.reference_parameters_
            rel_ref_paras = test_ds.reference_parameters_relative_to_wb_
            for p in (ref_paras, rel_ref_paras):
                ref_gs = p.wb_list
                assert len(ref_gs) == len(test.raw_reference_parameters[reference_para_level])
                assert len(p.ic_list) == len(
                    pd.concat(
                        [
                            pd.Series(wb["InitialContact_Event"])
                            for wb in test.raw_reference_parameters[reference_para_level]
                        ]
                    ).dropna()
                )

    def test_duplicated_metadata(self):
        ds = GenericMobilisedDataset(
            sorted([p / "data.mat" for p in get_all_lab_example_data_paths().values()]),
            GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_lab"],
            # With these setting, we will get duplicated metadata, because just the p_id is not unique
            (None, "participant_id"),
            measurement_condition="laboratory",
        )

        with pytest.raises(ValueError, match="The metadata for each file path must be unique."):
            _ = ds.index

    def test_no_metadata_for_multiple_files(self):
        ds = GenericMobilisedDataset(
            sorted([p / "data.mat" for p in get_all_lab_example_data_paths().values()]),
            GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_lab"],
            measurement_condition="laboratory",
            # With these setting, we will get duplicated metadata, because just the p_id is not unique
        )

        with pytest.raises(ValueError, match="It seems like no metadata for the files was provided."):
            _ = ds.index

    def test_invalid_path_type(self):
        ds = GenericMobilisedDataset(
            (PROJECT_ROOT / "example_data/data/lab").rglob("data.mat"),
            test_level_names=GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_lab"],
            measurement_condition="laboratory",
        )

        with pytest.raises(TypeError, match="paths_list must be a PathLike or a Sequence of PathLikes"):
            _ = ds.index

    def test_missing_sensor_position_default(self):
        ds = GenericMobilisedDataset(
            sorted([p / "data.mat" for p in get_all_lab_example_data_paths().values()]),
            GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_lab"],
            ("cohort", "participant_id"),
            sensor_positions=("UnkownSensor",),
            measurement_condition="laboratory",
        )

        with pytest.raises(
            ValueError,
        ) as e:
            _ = ds[0].data_ss

        assert "Expected sensor data for {('SU', 'UnkownSensor')}" in str(e.value)

    def test_missing_sensor_position_raise(self):
        ds = GenericMobilisedDataset(
            sorted([p / "data.mat" for p in get_all_lab_example_data_paths().values()]),
            GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_lab"],
            ("cohort", "participant_id"),
            sensor_positions=("UnkownSensor",),
            missing_sensor_error_type="raise",
            measurement_condition="laboratory",
        )

        with pytest.raises(
            ValueError,
        ) as e:
            _ = ds[0].data_ss

        assert "Expected sensor data for {('SU', 'UnkownSensor')}" in str(e.value)

    def test_missing_sensor_position_warn(self):
        ds = GenericMobilisedDataset(
            sorted([p / "data.mat" for p in get_all_lab_example_data_paths().values()]),
            GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_lab"],
            ("cohort", "participant_id"),
            sensor_positions=("UnkownSensor",),
            missing_sensor_error_type="warn",
            measurement_condition="laboratory",
        )

        with pytest.warns(
            UserWarning,
            match=r"Expected sensor data for {\('SU', 'UnkownSensor'\)}",
        ):
            _ = ds[0].data

    def test_error_missing_sensor_default(self, example_missing_data_path):
        """Test missing sensor data for default setting"""
        # Test default loading
        with pytest.raises(
            ValueError,
            match=r"Expected sensor data for {\('SU', 'LowerBack'\)}",
        ):
            _ = load_mobilised_matlab_format(example_missing_data_path / "data.mat", sensor_positions=("LowerBack",))

    def test_error_missing_sensor_warn(self, example_missing_data_path):
        """Test missing sensor data for missing_sensor_error_type='warn'"""
        with pytest.warns(UserWarning) as w:
            result = load_mobilised_matlab_format(
                example_missing_data_path / "data.mat",
                sensor_positions=("LowerBack",),
                missing_sensor_error_type="warn",
            )

        assert len(w) == 1

        assert "Expected sensor data for {('SU', 'LowerBack')}" in str(w[0].message)

        assert result[("TimeMeasure1", "Test11", "Trial1")].imu_data == {}
        assert result[("TimeMeasure1", "Test11", "Trial1")].metadata["sampling_rate_hz"] is None

    def test_error_missing_sensor_ignore(self, example_missing_data_path):
        """Test missing sensor data for missing_sensor_error_type='ignore'. No Warning should be emitted"""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = load_mobilised_matlab_format(
                example_missing_data_path / "data.mat",
                sensor_positions=("LowerBack",),
                missing_sensor_error_type="ignore",
            )

        assert result[("TimeMeasure1", "Test11", "Trial1")].imu_data == {}
        assert result[("TimeMeasure1", "Test11", "Trial1")].metadata["sampling_rate_hz"] is None
