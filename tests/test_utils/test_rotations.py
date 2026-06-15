import pandas as pd
from pandas._testing import assert_frame_equal
from scipy.spatial.transform import Rotation

from mobgap._gaitmap.utils.rotations import flip_dataset


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
