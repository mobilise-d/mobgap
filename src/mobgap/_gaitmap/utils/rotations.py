from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from mobgap.consts import SF_ACC_COLS, SF_GYR_COLS


def _rotate_sensor(data: pd.DataFrame, rotation: Optional[Rotation]) -> pd.DataFrame:
    """Rotate the data of a single sensor with acc and gyro."""
    data = data.copy()
    if rotation is None:
        return data
    data[SF_GYR_COLS] = rotation.apply(data[SF_GYR_COLS].to_numpy(copy=True))
    data[SF_ACC_COLS] = rotation.apply(data[SF_ACC_COLS].to_numpy(copy=True))
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


def rotation_from_angle(axis: np.ndarray, angle: Union[float, np.ndarray]) -> Rotation:
    """Create a rotation based on a rotation axis and a angle.

    Parameters
    ----------
    axis : array with shape (3,) or (n, 3)
        normalized rotation axis ([x, y ,z]) or array of rotation axis
    angle : float or array with shape (n,)
        rotation angle or array of angeles in rad

    Returns
    -------
    rotation(s) : Rotation object with len n

    Examples
    --------
    Single rotation: 180 deg rotation around the x-axis

    >>> rot = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(180))
    >>> rot.as_quat().round(decimals=3)
    array([1., 0., 0., 0.])
    >>> rot.apply(np.array([[0, 0, 1.0], [0, 1, 0.0]])).round()
    array([[ 0., -0., -1.],
           [ 0., -1.,  0.]])

    Multiple rotations: 90 and 180 deg rotation around the x-axis

    >>> rot = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad([90, 180]))
    >>> rot.as_quat().round(decimals=3)
    array([[0.707, 0.   , 0.   , 0.707],
           [1.   , 0.   , 0.   , 0.   ]])
    >>> # In case of multiple rotations, the first rotation is applied to the first vector
    >>> # and the second to the second
    >>> rot.apply(np.array([[0, 0, 1.0], [0, 1, 0.0]])).round()
    array([[ 0., -1.,  0.],
           [ 0., -1.,  0.]])

    """
    angle = np.atleast_2d(angle)
    axis = np.atleast_2d(axis)
    return Rotation.from_rotvec(np.squeeze(axis * angle.T))
