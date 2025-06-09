"""Some basic conversion utilities."""

from collections.abc import Sequence
from typing import TypeVar, Union, overload

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from mobgap._gaitmap.utils.rotations import rotate_dataset_series
from mobgap.consts import COLS_PER_FRAME, GF_SENSOR_COLS, SF_SENSOR_COLS
from mobgap.utils.dtypes import get_frame_definition

T = TypeVar("T", float, int, Sequence[float], Sequence[int])


@overload
def as_samples(sec_value: Union[int, float], sampling_rate_hz: float) -> int: ...


@overload
def as_samples(sec_value: Union[Sequence[int], Sequence[float]], sampling_rate_hz: float) -> Sequence[int]: ...


@overload
def as_samples(sec_value: np.ndarray, sampling_rate_hz: float) -> np.ndarray: ...


def as_samples(sec_value, sampling_rate_hz):
    """Convert seconds to samples.

    Parameters
    ----------
    sec_value
        The value in seconds.
    sampling_rate_hz
        The sampling rate in Hertz.

    Returns
    -------
    converted_samples
        The value in samples.

    """
    if isinstance(sec_value, np.ndarray):
        return np.round(sec_value * sampling_rate_hz).astype("int64")
    if isinstance(sec_value, (int, float)):
        return int(np.round(sec_value * sampling_rate_hz))
    if isinstance(sec_value, (pd.DataFrame, pd.Series)):
        return (sec_value * sampling_rate_hz).round().astype("int64")
    return type(sec_value)(int(np.round(s * sampling_rate_hz)) for s in sec_value)


def transform_to_global_frame(data: pd.DataFrame, orientations: Rotation) -> pd.DataFrame:
    """Transform the data to the global frame using the given rotations.

    Depending on if the data is in the sensor frame or the body frame, the output will be transformed to the normal
    global frame (gx, gy, gz) or the body frame aligned global frame (gis, gml, gpa).

    .. note:: The rotations need to be defined based on the normal global frame in mobgap and not based on the
       local sensor/body frame. So the rotaitons should define the transformation from the sensor/body frame to the
       normal global frame.

    Parameters
    ----------
    data
        The data to transform.
    orientations
        The global frame orientations estimated by the orientation estimation algorithm.

    Returns
    -------
    transformed_data
        The transformed data.

    """
    if not len(data) == len(orientations):
        raise ValueError(
            "The data and the orientations need to have the same length. "
            "Some orientation methods return n+1 orientations. "
            "In this case, use `orientations[:-1]`."
        )
    frame = get_frame_definition(data, ["sensor", "body"])
    if frame == "body":
        # If the data was provided in body frame coordinates, we need to convert it to the sensor frame first
        # because the rotation function expects this to correctly define the internal vectors.
        data = to_sensor_frame(data)
    rotated_data = rotate_dataset_series(data, orientations)

    # We know that the data is now in the global frame.
    # So we rename the columns to match the global frame axis names.
    rotated_data = rotated_data.rename(columns=dict(zip(SF_SENSOR_COLS, GF_SENSOR_COLS)))

    if frame == "body":
        # If the data was originally in the body frame, we want to have it in the body aligned global frame.
        # This requires a renaming of the columns and an axis flip.
        return to_body_frame(rotated_data)
    return rotated_data


def to_body_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Rename the columns of the passed dataframe to match the body frame axis names.

    This will work for either data in the sensor frame or the global frame.
    Data from the sensor frame will be converted in the body-frame (x -> is, y -> ml, z -> pa).
    Data in the global frame will be converted to the body-frame aligned global frame
    (gx -> gpa, -gy -> gml, gz -> gis).
    In this case, the y-axis is inverted to match the body frame axis definitions and keep the coordinate system
    right-handed.

    In both cases, we assume that the coordinate system definitions of the provided data matches the mobgap/Mobilise-D
    guidelines.

    Parameters
    ----------
    data
        The dataframe to rename.

    Returns
    -------
    pd.DataFrame
        The dataframe with the columns renamed to match the body frame axis names

    """
    frame = get_frame_definition(data, ["sensor", "global"])

    conversions = {
        "sensor": dict(zip(COLS_PER_FRAME["sensor"], COLS_PER_FRAME["body"])),
        "global": {
            f"{sensor}_g{axis}": f"{sensor}_g{axis_new}"
            for axis, axis_new in (("z", "is"), ("y", "ml"), ("x", "pa"))
            for sensor in ("acc", "gyr")
        },
    }

    out_cols = COLS_PER_FRAME["body"] if frame == "sensor" else COLS_PER_FRAME["global_body"]

    renamed_df = data.rename(columns=conversions[frame])[out_cols]
    if frame == "global":
        renamed_df["acc_gml"] *= -1
        renamed_df["gyr_gml"] *= -1

    return renamed_df


def to_normal_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Rename the columns of the passed dataframe to match the normal global frame axis or the sensor frame axis names.

    This will work for either data in the body frame or the global body frame.
    Data from the body frame will be converted in the sensor frame (is -> x, ml -> y, pa -> z).
    Data in the body-aligned global frame will be converted to the normal global frame (gis -> gz, -gml -> y, gpa -> x).

    In both cases, we assume that the coordinate system definitions of the provided data matches the mobgap/Mobilise-D
    guidelines.

    Parameters
    ----------
    data
        The dataframe to rename.

    Returns
    -------
    pd.DataFrame
        The dataframe with the columns renamed to match the normal global frame axis names

    """
    frame = get_frame_definition(data, ["body", "global_body"])

    conversions = {
        "body": dict(zip(COLS_PER_FRAME["body"], COLS_PER_FRAME["sensor"])),
        "global_body": {
            f"{sensor}_g{axis}": f"{sensor}_g{axis_new}"
            for axis, axis_new in (("is", "z"), ("ml", "y"), ("pa", "x"))
            for sensor in ("acc", "gyr")
        },
    }

    out_cols = COLS_PER_FRAME["global"] if frame == "global_body" else COLS_PER_FRAME["sensor"]

    renamed_df = data.rename(columns=conversions[frame])[out_cols]

    if frame == "global_body":
        renamed_df["acc_gy"] *= -1
        renamed_df["gyr_gy"] *= -1

    return renamed_df


to_sensor_frame = to_normal_frame


__all__ = ["as_samples", "to_body_frame", "to_normal_frame", "to_sensor_frame", "transform_to_global_frame"]
