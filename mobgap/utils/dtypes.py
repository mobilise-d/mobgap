"""Helper to validate and convert common data types used in mobgap."""

from typing import Any, Callable, Literal, Optional, TypeVar, Union

import numpy as np
import pandas as pd
from typing_extensions import TypeAlias

from mobgap.consts import COLS_PER_FRAME

# : Type alias for dataframe-like objects.
DfLike: TypeAlias = Union[pd.Series, pd.DataFrame, np.ndarray]
# : The type variable for dataframe-like objects.
DfLikeT = TypeVar("DfLikeT", bound=DfLike)


def is_dflike(data: Any) -> bool:
    """Check if the passed data is dataframe-like.

    This includes pandas dataframes and series, as well as numpy arrays.

    Parameters
    ----------
    data
        The data to check.

    Returns
    -------
    bool
        Whether the passed data is dataframe-like.

    """
    return isinstance(data, (pd.Series, pd.DataFrame, np.ndarray))


def dflike_as_2d_array(
    data: DfLikeT,
) -> tuple[np.ndarray, Optional[pd.Index], Callable[[np.ndarray, Optional[pd.Index]], DfLikeT]]:
    """Convert the passed data to a 2d numpy array and return a function to convert it back to the original datatype.

    We expect that each row in the data represents one timepoint and each column represents one sensor axis.
    The returned data is reshaped in a way that the first dimension represents the timepoints and the second dimension
    represents the sensor axes.

    This supports the following conversions:

    - ``pd.Series`` with length ``n``-> ``np.ndarray`` with shape ``(n, 1)``
    - ``pd.DataFrame`` with shape ``(m_cols, n_rows)`` -> ``np.ndarray`` with shape ``(n_rows, m_cols)``
    - ``np.ndarray`` with shape ``(n)`` -> ``np.ndarray`` with shape ``(n, 1)``
    - ``np.ndarray`` with shape ``(m, n)`` -> ``np.ndarray`` with shape ``(m, n)``

    Arrays with 0 or more than 2 dimensions are not supported.
    Conversion of ``pd.DataFrame`` and ``pd.Series`` objects will attempt to not copy the data and will preserve the
    index.

    The returned numpy array and index can be passed into the returned function to convert the data back to the original
    datatype.
    We return the index as well, to allow for potential manipulation of the index during the conversion.
    For example, when you want to resample the data, you can pass the index to the resampling function and then pass
    the resampled data and the resampled index to the returned function to convert the resampled data back to the same
    datatype as the original data.

    Parameters
    ----------
    data
        The data to convert.
        Must be dataframe-like (i.e. a dataframe, series, or numpy array).

    Returns
    -------
    np.ndarray
        The data as a 2d numpy array.
    Optional[pd.Index]
        The index of the passed data.
        This is only returned if the passed data is a ``pd.Series`` or ``pd.DataFrame``.
        Otherwise, ``None`` is returned.
    Callable[[np.ndarray, Optional[pd.Index]], DfLike]
        A function to convert the data back to the original datatype.
        This can be used to convert the results of a data transformation performed on the converted array back to the
        original datatype.
        Note, that these functions will attempt to not copy the data when converting back to a pandas object.

    """
    if not is_dflike(data):
        raise TypeError("The passed data is not dataframe-like (i.e. a dataframe, series, or numpy array).")

    if isinstance(data, np.ndarray):
        if data.ndim > 2 or data.ndim == 0:
            raise ValueError("The passed data must have 1 or 2 dimensions.")
        if data.ndim == 1:
            return data.reshape(-1, 1), None, lambda x, _: x.reshape(-1)
        return data, None, lambda x, _: x

    if isinstance(data, pd.Series):
        return (
            data.to_numpy(copy=False).reshape(-1, 1),
            data.index,
            lambda x, i: pd.Series(x.reshape(data.shape), index=i, copy=False, name=data.name),
        )

    return (
        data.to_numpy(copy=False),
        data.index,
        lambda x, i: pd.DataFrame(x, columns=data.columns, index=i, copy=False),
    )


def assert_is_sensor_data(data: pd.DataFrame, frame: Literal["sensor", "body", "global", "global_body"]) -> None:
    """Check if the passed dataframe contains sensor frame data.

    This is done by checking if the dataframe contains the columns defined in :obj:`~mobgap.consts.SF_SENSOR_COLS`.

    Parameters
    ----------
    data
        The dataframe to check.
    frame
        The frame to check for.
        Must be one of "sensor", "body", "global", or "global_body".
        Depending on the frame the dataframe must contain the corresponding columns.

    """
    if not isinstance(data, pd.DataFrame):
        raise AssertionError("The passed data is no valid imu data, as it is not a pandas dataframe.")  # noqa: TRY004
    try:
        expected_cols = COLS_PER_FRAME[frame]
    except KeyError as e:
        raise ValueError(f"Unknown frame {frame}. Must be one of {list(COLS_PER_FRAME.keys())}.") from e
    missing_cols = set(expected_cols) - set(data.columns)
    if missing_cols:
        raise AssertionError(
            f"The passed data is no valid imu data in the {frame} frame, as it is missing the following columns: "
            f"{missing_cols}."
        )


def get_frame_definition(
    data: pd.DataFrame, potential_frames: list[Literal["sensor", "body", "global", "global_body"]]
) -> Literal["sensor", "body", "global", "global_body"]:
    """Check if the passed dataframe contains sensor frame data of one of the potential frames definitions.

    Returns the first frame definition that matches the columns of the dataframe or raises an AssertionError
    if none of the potential frames match.
    """
    for frame in potential_frames:
        try:
            assert_is_sensor_data(data, frame)
        except AssertionError:
            continue
        return frame
    raise AssertionError(
        f"The passed data is no valid imu data, as it does not match any of the potential frames: {potential_frames}."
    )


__all__ = [
    "DfLike",
    "DfLikeT",
    "assert_is_sensor_data",
    "dflike_as_2d_array",
    "get_frame_definition",
    "is_dflike",
]
