import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from scipy.spatial.transform import Rotation

from mobgap.utils.rotations import flip_dataset


def test_flip_dataset_supports_sensor_frame_data():
    data = pd.DataFrame(
        {
            "acc_x": [1.0, 2.0],
            "acc_y": [3.0, 4.0],
            "acc_z": [5.0, 6.0],
            "gyr_x": [7.0, 8.0],
            "gyr_y": [9.0, 10.0],
            "gyr_z": [11.0, 12.0],
        }
    )

    result = flip_dataset(data, Rotation.from_euler("z", 90, degrees=True))

    expected = pd.DataFrame(
        {
            "acc_x": [-3.0, -4.0],
            "acc_y": [1.0, 2.0],
            "acc_z": [5.0, 6.0],
            "gyr_x": [-9.0, -10.0],
            "gyr_y": [7.0, 8.0],
            "gyr_z": [11.0, 12.0],
        }
    )
    assert_frame_equal(result, expected)


def test_flip_dataset_still_supports_body_frame_data():
    data = pd.DataFrame(
        {
            "acc_is": [1.0, 2.0],
            "acc_ml": [3.0, 4.0],
            "acc_pa": [5.0, 6.0],
            "gyr_is": [7.0, 8.0],
            "gyr_ml": [9.0, 10.0],
            "gyr_pa": [11.0, 12.0],
        }
    )

    result = flip_dataset(data, Rotation.from_euler("z", 90, degrees=True))

    expected = pd.DataFrame(
        {
            "acc_is": [-3.0, -4.0],
            "acc_ml": [1.0, 2.0],
            "acc_pa": [5.0, 6.0],
            "gyr_is": [-9.0, -10.0],
            "gyr_ml": [7.0, 8.0],
            "gyr_pa": [11.0, 12.0],
        }
    )
    assert_frame_equal(result, expected)


def test_flip_dataset_applies_one_rotation_to_all_samples_without_changing_input():
    data = pd.DataFrame(
        {
            "acc_x": [1.0, 2.0],
            "acc_y": [3.0, 4.0],
            "acc_z": [5.0, 6.0],
            "gyr_x": [7.0, 8.0],
            "gyr_y": [9.0, 10.0],
            "gyr_z": [11.0, 12.0],
            "sample_label": ["first", "second"],
        },
        index=pd.Index([10, 20], name="sample"),
    )
    original = data.copy(deep=True)

    result = flip_dataset(data, Rotation.from_euler("x", 180, degrees=True))

    expected = data.copy()
    expected[["acc_y", "acc_z", "gyr_y", "gyr_z"]] *= -1
    assert_frame_equal(result, expected)
    assert_frame_equal(data, original)


def test_flip_dataset_none_returns_independent_copy():
    data = pd.DataFrame({col: [1.0] for col in ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")})

    result = flip_dataset(data, None)

    assert_frame_equal(result, data)
    assert result is not data


@pytest.mark.parametrize("rotation", [Rotation.from_euler("z", 45, degrees=True), Rotation.identity(2)])
def test_flip_dataset_rejects_rotations_that_cannot_be_applied_as_axis_flips(rotation):
    data = pd.DataFrame({col: [1.0] for col in ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")})

    with pytest.raises(ValueError, match="Only"):
        flip_dataset(data, rotation)
