from pathlib import Path

import joblib
import pytest

from gaitlink.data import (
    GenericMobilisedDataset,
    load_mobilised_matlab_format,
    load_mobilised_participant_metadata_file,
)
from gaitlink.data import get_all_lab_example_data_paths, load_mobilised_matlab_format


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
    assert t1_metadata["Height"] == 176
    assert t1_metadata["Handedness"] == "R"
    assert t1_metadata["SensorType_SU"] == "MM+"


class TestDatasetClass:
    def test_index_creation(self, example_data_path):
        ds = GenericMobilisedDataset(
            list(Path("/home/arne/Downloads/Mobilise-D dataset_1-18-2023").rglob("data.mat")),
            ("time_measure", "recording"),
            ("cohort",),
            memory=joblib.Memory(".cache"),
        )

        ds.index
        print(ds.index)

        # ds = GenericMobilisedDataset(
        #     list(Path(example_data_path).rglob("data.mat")),
        #     GenericMobilisedDataset.COMMON_TEST_LEVEL_NAMES["tvs_lab"],
        #     ("cohort",),
        #     memory=joblib.Memory(".cache"),
        # )

        data = ds[3].data
        print(data)
        print(ds[3].participant_metadata)
        print(ds[3].metadata)
