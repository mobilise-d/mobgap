"""Some basic conversion utilities."""

from collections.abc import Sequence
from typing import TypeVar, Union, overload

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from mobgap._gaitmap.utils.rotations import rotate_dataset_series
from mobgap.consts import GF_SENSOR_COLS, SF_SENSOR_COLS
from mobgap.utils.dtypes import get_frame_definition, to_body_frame, to_sensor_frame

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
    return type(sec_value)(int(np.round(s * sampling_rate_hz)) for s in sec_value)


def transform_to_global_frame(data: pd.DataFrame, orientations: Rotation) -> pd.DataFrame:
    """Transform the data to the global frame using the given rotations.

    Depending on if the data is in the sensor frame or the body frame, the output will be transformed to the normal
    global frame (gx, gy, gz) or the body frame aligned global frame (gis, gml, gpa).

    .. note:: The rotations need to be defined based on the normal global frame in mobgap and not based on the
       local sensor/body frame.

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


__all__ = ["as_samples"]
