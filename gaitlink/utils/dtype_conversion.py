from typing import Any, Callable, TypeVar, Union

import numpy as np
import pandas as pd
from typing_extensions import TypeAlias

DfLike: TypeAlias = Union[pd.Series, pd.DataFrame, np.ndarray]
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


def dflike_as_2d_array(data: DfLikeT) -> tuple[np.ndarray, Callable[[np.ndarray], DfLikeT]]:
    """Convert the passed data to a 2d numpy array and return a function to convert it back to the original datatype.

    This supports the following conversions:

    - ``pd.Series`` with length ``n``-> ``np.ndarray`` with shape ``(1, n)``
    - ``pd.DataFrame`` with shape ``(m, n)`` -> ``np.ndarray`` with shape ``(m, n)``
    - ``np.ndarray`` with shape ``(n)`` -> ``np.ndarray`` with shape ``(1, n)``
    - ``np.ndarray`` with shape ``(m, n)`` -> ``np.ndarray`` with shape ``(m, n)``

    Arrays with 0 or more than 2 dimensions are not supported.
    Conversion of ``pd.DataFrame`` and ``pd.Series`` objects will attempt to not copy the data and will preserve the
    index.


    Parameters
    ----------
    data
        The data to convert.
        Must be dataframe-like (i.e. a dataframe, series, or numpy array).

    Returns
    -------
    np.ndarray
        The data as a 2d numpy array.
    Callable[[np.ndarray], DfLike]
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
            return data.reshape(1, -1), lambda x: x.reshape(-1)
        return data, lambda x: x

    if isinstance(data, pd.Series):
        return data.to_numpy().reshape(1, -1), lambda x: pd.Series(
            x.reshape(data.shape), index=data.index, copy=False, name=data.name
        )

    return data.to_numpy(), lambda x: pd.DataFrame(x, columns=data.columns, index=data.index, copy=False)
