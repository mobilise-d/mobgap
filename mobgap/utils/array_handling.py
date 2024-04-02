"""Utility functions to perform common array operations."""

from collections.abc import Hashable, Iterator
from typing import Any, Callable, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as np_sliding_window_view
from typing_extensions import Unpack


def sliding_window_view(data: np.ndarray, window_size_samples: int, overlap_samples: int) -> np.ndarray:
    """Create a sliding window view of the data.

    Note, the output will be a view of the data, not a copy.
    This makes it more efficient, but also means, that the data should not be modified.

    If the length of the data can not be divided by the window size, remaining samples will be dropped.

    Parameters
    ----------
    data
        The data to create the sliding window view of.
        This data can be n-dimensional.
        However, the window will only be applied to the first axis (axis=0).
        See Notes for details.
    window_size_samples
        The size of the sliding window in samples.
    overlap_samples
        The overlap of the sliding window in samples.

    Returns
    -------
    np.ndarray
        The sliding window view of the data.

    Notes
    -----
    In case of nd-arrays, the output format looks as follows:

    Assume the input array has the shape (I, J, K) and we want to create a sliding window view with a window size of N
    and no overlap.
    Then the output array will have the shape (I // N, N, J, K).

    This is different from the output of :func:`numpy.lib.stride_tricks.sliding_window_view`, which would have the shape
    (I // N, J, K, N).
    We changed this here, so that when inspecting each window, you would still have the expected array shape, assuming
    that your original first axis was time.

    """
    if overlap_samples > window_size_samples:
        raise ValueError("overlap_samples must be smaller than window_size_samples")

    view = np_sliding_window_view(data, window_shape=(window_size_samples,), axis=0)[
        :: (window_size_samples - overlap_samples)
    ]

    if data.ndim > 1:
        view = np.moveaxis(view, -1, 1)

    return view


def _loc_with_empty_fallback(df: pd.DataFrame, name: Any) -> pd.DataFrame:
    try:
        return df.loc[name]
    except KeyError:
        # We return a frame that has the same columns as the original, but no rows.
        # We also replicate the index, just without any rows.

        index = df.index[:0]
        return pd.DataFrame(columns=df.columns, index=index)


class MultiGroupBy:
    """Object representing the grouping result of multiple dataframes.

    This is used as proxy object to replicate an API similar to the normal pandas groupy object, but allowing
    to group multiple dataframes by the same index levels to apply a function to each group across all dataframes.

    See :func:`~create_multi_groupby` for the creation of this object.

    """

    _primary_groupby: pd.core.groupby.DataFrameGroupBy

    def __init__(
        self,
        primary_df: pd.DataFrame,
        secondary_dfs: Union[pd.DataFrame, list[pd.DataFrame]],
        groupby: Union[str, list[str]],
    ) -> None:
        self.primary_df = primary_df
        self.secondary_dfs = secondary_dfs
        if not isinstance(secondary_dfs, list):
            self.secondary_dfs = [secondary_dfs]
        self.groupby = groupby
        if isinstance(groupby, str):
            self.groupby = [groupby]

        # For the approach to work, all dfs need to have the same index columns with the `level` columns as the first
        # index levels.
        primary_index_cols = primary_df.index.names
        if not set(self.groupby).issubset(primary_index_cols):
            raise ValueError("All `groupby` columns need to be in the index of all dataframes.")
        primary_index_cols_reorderd = [
            *self.groupby,
            *[col for col in primary_index_cols if col not in groupby],
        ]

        self.primary_df = primary_df.reorder_levels(primary_index_cols_reorderd)
        self.secondary_dfs = [df.reorder_levels(primary_index_cols_reorderd) for df in self.secondary_dfs]

    @property
    def primary_groupby(self) -> pd.core.groupby.DataFrameGroupBy:
        """The primary groupby object.

        This is the grouper created from the primary dataframe.
        """
        if not hasattr(self, "_primary_groupby"):
            self._primary_groupby = self.primary_df.groupby(level=self.groupby)
        return self._primary_groupby

    def _get_secondary_vals(self, name: Union[str, tuple[str, ...]]) -> list[pd.DataFrame]:
        return [_loc_with_empty_fallback(df, [name]) for df in self.secondary_dfs]

    def get_group(self, name: Union[str, tuple[str, ...]]) -> tuple[pd.DataFrame, ...]:
        """Get an individual group by name.

        Returns
        -------
        A tuple containing the groups from each dataframe.
        """
        return_val = self.primary_groupby.get_group(name), *self._get_secondary_vals(name)
        return return_val

    @property
    def groups(
        self,
    ) -> dict[Union[str, tuple[str, ...]], tuple[pd.DataFrame, ...]]:
        """A dictionary with the group names as keys and the group dataframes as values."""
        return dict(self)

    @property
    def ngroups(self) -> int:
        """The number of groups."""
        return self.primary_groupby.ngroups

    def __len__(self) -> int:
        """Get the number of groups."""
        return len(self.primary_groupby)

    def __iter__(
        self,
    ) -> Iterator[tuple[Union[str, tuple[str, ...]], tuple[pd.DataFrame, ...]]]:
        """Iterate over the groups and return a tuple with the group name and the group dataframes."""
        return ((name, self.get_group(name)) for name, group in self.primary_groupby)

    def apply(self, func: Callable, *args: Unpack[list[Any]], **kwargs: Unpack[dict[str, Any]]) -> pd.DataFrame:
        """Apply a function that takes the group values from each df as input.

        The function is expected to take n dataframes as input, where n is the number of secondary dataframes + 1.
        The expected signature is:
        ``func(group_df_prim, group_df_sec_0, group_df_sec_1, ..., *args, **kwargs)``

        """

        def _nested_func(group: pd.DataFrame, *iargs: Any, **ikwargs: Unpack[dict[str, Any]]) -> Any:
            secondary_vals = self._get_secondary_vals(group.name)
            return func(group, *secondary_vals, *iargs, **ikwargs)

        return self.primary_groupby.apply(_nested_func, *args, **kwargs)


def create_multi_groupby(
    primary_df: pd.DataFrame,
    secondary_dfs: Union[pd.DataFrame, list[pd.DataFrame]],
    groupby: Union[Hashable, list[str]],
) -> MultiGroupBy:
    """Group multiple dataframes by the same index levels to apply a function to each group across all dataframes.

    This function will return an object similar to a :class:`~pandas.core.groupby.DataFrameGroupBy` object, but with
    only the ``apply`` and ``__iter__`` methods implemented.
    This special groupby object applies a groupby to the primary dataframe, but when iterating over the groups, or
    applying a function, it will also provide the groups of the secondary dataframes by using ``loc`` with the group
    name of the primary dataframe.

    This also means that this function is much more limited than the standard groupby object, as it only supports the
    grouping by existing **named** index levels and forces all dataframes to have the same index columns.

    .. warning:: It is important to understand that we only groupy the index of the primary dataframe!
       This means if an index value only exists in one of the secondary dataframes, it will be ignored.
       We do this to be able to "just" use the normal pandas groupby API under the hood.
       We simply group the primary dataframe, get the corresponding groups from the secondary dataframes (if available)
       and inject them into all operations.

    Parameters
    ----------
    primary_df
        The primary dataframe to group by.
        Its index will be used to perform the actual grouping.
    secondary_dfs
        The secondary dataframes to group by.
    groupby
        The names of the index levels to group by.

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {
    ...         "group1": [1, 1, 2, 3],
    ...         "group2": [1, 2, 1, 1],
    ...         "value": [1, 2, 3, 4],
    ...     }
    ... ).set_index(["group1", "group2"])
    >>> df_2 = pd.DataFrame(
    ...     {
    ...         "group1": [1, 1, 1, 2],
    ...         "group2": [1, 2, 3, 1],
    ...         "value": [11, 12, 13, 14],
    ...     }
    ... ).set_index(["group1", "group2"])
    >>> multi_groupby = create_multi_groupby(df, df_2, ["group1"])
    >>> for group, (df1, df2) in multi_groupby:
    ...     print(group)
    ...     print(df1)
    ...     print(df2)
    1
                   value
    group1 group2
    1      1           1
           2           2
                   value
    group1 group2
    1      1          11
           2          12
           3          13
    2
                   value
    group1 group2
    2      1           3
                   value
    group1 group2
    2      1          14
    3
                   value
    group1 group2
    3      1           4
    Empty DataFrame
    Columns: [value]
    Index: []

    """
    return MultiGroupBy(primary_df, secondary_dfs, groupby)


__all__ = ["sliding_window_view", "create_multi_groupby", "MultiGroupBy"]
