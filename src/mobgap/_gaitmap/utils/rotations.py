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


def _flip_sensor(data: pd.DataFrame, rotation: Optional[Rotation]) -> pd.DataFrame:
    """Flip (same as rotate, but only 90 deg rots allowed) the data of a single sensor.

    Compared to normal rotations, this function can result in massive speedups!
    """
    if rotation.single is False:
        raise ValueError("Only single rotations are allowed!")

    tol = 10e-9
    rot_matrix = rotation.as_matrix().squeeze()
    all_1 = np.allclose(np.abs(rot_matrix[~np.isclose(rot_matrix, 0, atol=tol)]).flatten(), 1, atol=tol)
    if not all_1:
        raise ValueError(
            "Only 90 deg rotations are allowed (i.e. 1 and -1 in the rotation matrix)! "
            f"The current matrix is:\n\n {rot_matrix}"
        )

    # Now that we know the rotation is valid, we round the values to make all further checks simpler
    rot_matrix = np.round(rot_matrix)

    data = data.copy()
    if rotation is None:
        return data

    orig_col_order = data.columns
    sensors = ["acc", "gyr"]
    rots = {"acc": SF_ACC_COLS, "gyr": SF_GYR_COLS}
    for sensor in sensors:
        cols = np.array(rots[sensor])
        rename = {}
        mirror = []
        # We basically iterate over the rotation matrix and find which axis is transformed to which other axis.
        # If the entry is -1, we also mirror the axis.
        for col, row in zip(cols, rot_matrix):
            old_index = cols[np.abs(row).astype(bool)][0]
            rename[old_index] = col
            if np.sum(row) == -1:
                mirror.append(col)
        # We use inplace here to make sure we honor the inplace passed to this function
        data = data.rename(columns=rename)
        data[mirror] *= -1
    data = data[orig_col_order]
    return data


def flip_dataset(dataset: pd.DataFrame, rotation: Union[Rotation, dict[str, Rotation]]) -> pd.DataFrame:
    """Flip datasets around axis data of a dataset.

    This is equivalent to rotating the data, but only 90/180 deg rotations are allowed.
    With this restriction, we don't need to actually rotate the data, but can just swap and flip the columns.

    This method should be used, when roughly aligning the data to a reference frame, where you usually would only
    apply 90 deg rotations based on the known rough orientation of the sensor.
    If you need to apply arbitrary rotations, use `rotate_dataset` instead.

    Parameters
    ----------
    dataset
        dataframe representing a single or multiple sensors.
        In case of multiple sensors a df with MultiIndex columns is expected where the first level is the sensor name
        and the second level the axis names (all sensor frame axis must be present)
    rotation
        A single rotation or a dict with sensor names as keys and rotations as values.
        All rotations must only contain 90 deg rotations (i.e. 1 and -1 in the rotation matrix).
        If this is not the case, use :func:`rotate_dataset` instead.

    Returns
    -------
    flipped dataset
        This will always be a copy. The original dataframe will not be modified.


    See Also
    --------
    gaitmap.utils.rotations.rotate_dataset: Freely rotate a dataset

    """
    return _flip_sensor(dataset, rotation)
