"""Utility functions to handle rotations."""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from mobgap._gaitmap.utils.rotations import rotate_dataset_series
from mobgap.consts import BF_ACC_COLS, BF_GYR_COLS, SF_ACC_COLS, SF_GYR_COLS
from mobgap.utils.dtypes import get_frame_definition


def flip_dataset(dataset: pd.DataFrame, rotation: Optional[Rotation]) -> pd.DataFrame:
    """Apply a right-angle rotation to every sample of a single-sensor dataset.

    The rotation swaps and changes the signs of axes, so the original sample values
    are preserved exactly. Both sensor and body frame data are supported.

    Parameters
    ----------
    dataset
        Dataframe with acceleration and gyroscope columns in the sensor or body frame.
    rotation
        A single SciPy rotation whose matrix contains only 0, 1, and -1 (within
        floating-point tolerance). Use ``None`` to return a copy without rotation.

    Returns
    -------
    pd.DataFrame
        A copy with the rotation applied to acceleration and gyroscope data.
        The index, column order, and any other columns are preserved.

    """
    result = dataset.copy()
    if rotation is None:
        return result
    if not rotation.single:
        raise ValueError("Only a single rotation is allowed.")

    matrix = rotation.as_matrix()
    if not np.allclose(matrix, np.round(matrix), atol=1e-8):
        raise ValueError("Only rotations in 90-degree steps around the axes are allowed.")
    matrix = np.round(matrix).astype(int)

    frame = get_frame_definition(dataset, ["sensor", "body"])
    column_groups = (SF_ACC_COLS, SF_GYR_COLS) if frame == "sensor" else (BF_ACC_COLS, BF_GYR_COLS)
    for columns in column_groups:
        for output_axis, row in enumerate(matrix):
            source_axis = np.flatnonzero(row)[0]
            result[columns[output_axis]] = dataset[columns[source_axis]] * row[source_axis]
    return result


__all__ = ["flip_dataset", "rotate_dataset_series"]
