"""Utility functions to perform common array operations."""

from collections.abc import Iterator
from typing import Callable, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as np_sliding_window_view


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


class MultiGroupBy:
    _primary_groupby: pd.core.groupby.DataFrameGroupBy

    def __init__(
        self,
        primary_df: pd.DataFrame,
        secondary_dfs: Union[pd.DataFrame, list[pd.DataFrame]],
        index_level_names: Union[str, list[str]],
            *,
            drop_groupby_cols: bool = True,
    ) -> None:
        self.primary_df = primary_df
        self.secondary_dfs = secondary_dfs
        if not isinstance(secondary_dfs, list):
            self.secondary_dfs = [secondary_dfs]
        self.index_level_names = index_level_names
        if isinstance(index_level_names, str):
            self.index_level_names = [index_level_names]

        self.drop_groupby_cols = drop_groupby_cols

        # For the approach to work, all dfs need to have the same index columns with the `level` columns as the first
        # index levels.
        primary_index_cols = primary_df.index.names
        if not set(index_level_names).issubset(primary_index_cols):
            raise ValueError("All index_level_names need to be in the index of all dataframes.")
        primary_index_cols_reorderd = [
            *index_level_names,
            *[col for col in primary_index_cols if col not in index_level_names],
        ]

        self.primary_df = primary_df.reorder_levels(primary_index_cols_reorderd)
        self.secondary_dfs = [df.reorder_levels(primary_index_cols_reorderd) for df in secondary_dfs]

    @property
    def primary_groupby(self) -> pd.core.groupby.DataFrameGroupBy:
        if not hasattr(self, "_primary_groupby"):
            self._primary_groupby = self.primary_df.groupby(level=self.index_level_names)
        return self._primary_groupby

    def _get_secondary_vals(self, name: Union[str, tuple[str, ...]]) -> list[pd.DataFrame]:
        return [df.loc[[name]] for df in self.secondary_dfs]

    def get_group(self, name: Union[str, tuple[str, ...]]) -> tuple[pd.DataFrame, ...]:
        return_val =  self.primary_groupby.get_group(name), *self._get_secondary_vals(name)
        if not self.drop_groupby_cols:
            return return_val
        return tuple(df.droplevel(self.index_level_names) for df in return_val)

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
        """The number of groups."""
        return len(self.primary_groupby)

    def __iter__(
        self,
    ) -> Iterator[tuple[Union[str, tuple[str, ...]], tuple[pd.DataFrame, ...]]]:
        """Iterate over the groups and return a tuple with the group name and the group dataframes."""
        return ((name, self.get_group(name)) for name, group in self.primary_groupby)

    def apply(self, func: Callable, *args, **kwargs):
        """Apply a function that takes the group values from each df as input.

        The function is expected to take n dataframes as input, where n is the number of secondary dataframes + 1.
        The expected signature is:
        ``func(group_df_prim, group_df_sec_0, group_df_sec_1, ..., *args, **kwargs)``

        """

        def nested_func(group: pd.DataFrame, *args, **kwargs):
            secondary_vals = self._get_secondary_vals(group.name)
            return func(group, *secondary_vals, *args, **kwargs)

        return self.primary_groupby.apply(nested_func, *args, **kwargs)


def create_multi_groupby(
    primary_df: pd.DataFrame,
    secondary_dfs: Union[pd.DataFrame, list[pd.DataFrame]],
    index_level_names: Union[str, list[str]],
) -> MultiGroupBy:
    """Group multiple dataframes by the same index levels to apply a function to each group across all dataframes.

    This function will return an object similar to a :class:`~pandas.core.groupby.DataFrameGroupBy` object, but with
    only the ``apply`` and ``__iter__`` methods implemented.
    This special groupby object applies a groupby to the primary dataframe, but when iterating over the groups, or
    applying a function, it will also provide the groups of the secondary dataframes by using ``loc`` with the group
    name of the primary dataframe.

    This also means that this function is much more limited than the standard groupby object, as it only supports the
    grouping by existing index levels and forces all dataframes to have the same index columns.

    Parameters
    ----------
    primary_df
        The primary dataframe to group by.
    secondary_dfs
        The secondary dataframes to group by.
    index_level_names
        The names of the index levels to group by.
        If not provided, the first index level will be used.
    """
    return MultiGroupBy(primary_df, secondary_dfs, index_level_names)


__all__ = ["sliding_window_view", "create_multi_groupby"]
