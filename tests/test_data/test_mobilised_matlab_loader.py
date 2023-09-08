import pytest

from gaitlink.data import get_lab_example_data_path, load_mobilised_matlab_format


@pytest.fixture()
def example_file_path():
    return get_lab_example_data_path("HA", "001") / "data.mat"


def test_simple_file_loading(example_file_path, recwarn):
    data = load_mobilised_matlab_format(example_file_path)

    # We don't expect any user-warnings to be raised
    assert len([w for w in recwarn if issubclass(w.category, UserWarning)]) == 0

    assert len(data) == 3

    for name, test_data in data.items():
        assert isinstance(name, tuple)
        assert len(name) == 3

        # Test basic metadata
        # TODO: Change values once decided on actual example data
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

        # By default, there should be no reference parameter
        assert test_data.reference_parameters is None
        assert test_data.metadata.reference_sampling_rate_hz is None


def test_reference_system_loading(example_file_path):
    data = load_mobilised_matlab_format(example_file_path, reference_system="INDIP")

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
