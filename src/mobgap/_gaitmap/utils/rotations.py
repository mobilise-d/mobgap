from typing import Optional

import pandas as pd
from scipy.spatial.transform import Rotation

from mobgap.consts import SF_ACC_COLS, SF_GYR_COLS


def _rotate_sensor(data: pd.DataFrame, rotation: Optional[Rotation]) -> pd.DataFrame:
    """Rotate the data of a single sensor with acc and gyro."""
    data = data.copy()
    if rotation is None:
        return data
    data[SF_GYR_COLS] = rotation.apply(data[SF_GYR_COLS].to_numpy())
    data[SF_ACC_COLS] = rotation.apply(data[SF_ACC_COLS].to_numpy())
    return data


# Slighly modified from the original. We don't perform input datatype checks
def rotate_dataset_series(dataset: pd.DataFrame, rotations: Rotation) -> pd.DataFrame:
    """Rotate data of a single sensor using a series of rotations.

    This will apply a different rotation to each sample of the dataset.

    Parameters
    ----------
    dataset
        Data with axes names as ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"].
        If your data is defined in the body frame, use :func:`mobgap.utils.dtypes.to_sensor_frame` to convert it to the
        sensor frame first, as rotations are only defined for the sensor frame.
    rotations
        Rotation object that contains as many rotations as there are datapoints

    Returns
    -------
    rotated_data
        copy of `data` rotated by `rotations`

    """
    if len(dataset) != len(rotations):
        raise ValueError("The number of rotations must fit the number of samples in the dataset!")

    return _rotate_sensor(dataset, rotations)
