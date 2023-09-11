import pytest
from pandas._testing import assert_frame_equal

from gaitlink.data import (
    GenericMobilisedDataset,
    get_all_lab_example_data_paths,
    load_mobilised_matlab_format,
    load_mobilised_participant_metadata_file,
)


@pytest.fixture()
def example_data_path():
    return get_all_lab_example_data_paths()[("HA", "001")]


def test_simple_file_loading(example_data_path, recwarn, snapshot):
    data = load_mobilised_matlab_format(example_data_path / "data.mat")

    # We don't expect any user-warnings to be raised
    assert len([w for w in recwarn if issubclass(w.category, UserWarning)]) == 0

    assert len(data) == 3

    for name, test_data in data.items():
        assert isinstance(name, tuple)
        assert len(name) == 3

        # Test basic metadata
        assert test_data.metadata.time_zone == "Europe/Berlin"

        assert test_data.metadata.sampling_rate_hz == 100
        assert isinstance(test_data.metadata.start_date_time_iso, str)
        # First sample should be equivalent to the start date time
        assert (
            test_data.imu_data["LowerBack"]
            .index.tz_convert(test_data.metadata.time_zone)
            .round("ms")[0]
            .isoformat(timespec="milliseconds")
            == test_data.metadata.start_date_time_iso
        )

        # Test IMU data
        assert list(test_data.imu_data["LowerBack"].columns) == ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
        assert set(test_data.imu_data.keys()) == {"LowerBack"}
        assert len(test_data.imu_data["LowerBack"]) > 100

        snapshot_data = test_data.imu_data["LowerBack"].head(5)
        snapshot_data.index = snapshot_data.index.round("ms")
        snapshot.assert_match(snapshot_data, name)

        # By default, there should be no reference parameter
        assert test_data.reference_parameters is None
        assert test_data.metadata.reference_sampling_rate_hz is None


def test_reference_system_loading(example_data_path):
    data = load_mobilised_matlab_format(
        example_data_path / "data.mat",
        reference_system="INDIP",
    )

    number_of_tests_with_reference = 0

    for _name, test_data in data.items():
        if test_data.reference_parameters is not None:
            number_of_tests_with_reference += 1

            assert test_data.metadata.reference_sampling_rate_hz == 100
            assert set(test_data.reference_parameters.keys()) == {"lwb", "wb"}
            for _key, value in test_data.reference_parameters.items():
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


class TestDatasetClass:
    def test_index_creation(self, example_data_path):
        ds = GenericMobilisedDataset(
            sorted([p / "data.mat" for p in get_all_lab_example_data_paths().values()]),
            GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_lab"],
            ("cohort", "participant_id"),
        )

        manually_loaded_participant_index = list(get_all_lab_example_data_paths().keys())

        assert len(ds) == 3 * 3
        assert set(ds.index[["cohort", "participant_id"]].to_records(index=False).tolist()) == set(
            manually_loaded_participant_index
        )

        assert ds.index.columns.tolist() == ["cohort", "participant_id", "time_measure", "test", "trial"]

    def test_loaded_data(self, example_data_path):
        ds = GenericMobilisedDataset(
            example_data_path / "data.mat",
            GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_lab"],
            None,
            reference_system="INDIP",
        )

        manually_data = load_mobilised_matlab_format(example_data_path / "data.mat", reference_system="INDIP")

        assert list(ds.index.itertuples(index=False)) == list(manually_data.keys())

        for test_ds, test in zip(ds, manually_data.values()):
            assert test_ds.metadata == test.metadata
            assert test_ds.sampling_rate_hz == test.metadata.sampling_rate_hz
            assert test_ds.reference_sampling_rate_hz_ == test.metadata.reference_sampling_rate_hz
            assert test_ds.data.keys() == test.imu_data.keys()
            for sensor in test_ds.data:
                assert_frame_equal(test_ds.data[sensor], test.imu_data[sensor])
            # It is a little hard to compare the entire reference parameters, as they are a list of dicts
            # We just compare the length and the first entry
            for wb_type in test_ds.reference_parameters_:
                assert len(test_ds.reference_parameters_[wb_type]) == len(test.reference_parameters[wb_type])
                assert (
                    test_ds.reference_parameters_[wb_type][0]["Start"] == test.reference_parameters[wb_type][0]["Start"]
                )

    def test_duplicated_metadata(self):
        ds = GenericMobilisedDataset(
            sorted([p / "data.mat" for p in get_all_lab_example_data_paths().values()]),
            GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_lab"],
            # With these setting, we will get duplicated metadata, because just the p_id is not unique
            (None, "participant_id"),
        )

        with pytest.raises(ValueError, match="The metadata for each file path must be unique."):
            _ = ds.index
