"""Advnaced operations on complicated pandas dataframes."""

import warnings
from collections.abc import Hashable, Iterator, Sequence
from functools import wraps
from typing import Any, Callable, NamedTuple, Union

import numpy as np
import pandas as pd
from typing_extensions import Literal, Unpack


def _get_group_with_empty_fallback(
    group: pd.core.groupby.DataFrameGroupBy, original_df: pd.DataFrame, name: Any
) -> pd.DataFrame:
    try:
        return group.get_group(name)
    except KeyError:
        # We return a frame that has the same columns as the original, but no rows.
        # We also replicate the index, just without any rows.
        index = original_df.index[:0]
        return pd.DataFrame(columns=original_df.columns, index=index)


class MultiGroupByPrimaryDfEmptyError(Exception):
    """Error raised, when the primary df is empty."""


class MultiGroupBy:
    """Object representing the grouping result of multiple dataframes.

    This is used as proxy object to replicate an API similar to the normal pandas groupy object, but allowing
    to group multiple dataframes by the same index levels to apply a function to each group across all dataframes.

    See :func:`~create_multi_groupby` for the creation of this object.

    """

    _primary_groupby: pd.core.groupby.DataFrameGroupBy
    _secondary_groupbys: list[pd.core.groupby.DataFrameGroupBy]
    _kwargs: dict[str, Any]

    def __init__(
        self,
        primary_df: pd.DataFrame,
        secondary_dfs: Union[pd.DataFrame, list[pd.DataFrame]],
        groupby: Union[str, list[str]],
        **kwargs: Unpack[dict[str, Any]],
    ) -> None:
        if len(primary_df) == 0:
            raise MultiGroupByPrimaryDfEmptyError(
                "The primary df is empty and no groups could be identified. "
                "This error should be handled explicitly to decide what outputshape is desired in this case."
            )
        groupby_as_list = [groupby] if isinstance(groupby, str) else groupby
        self._kwargs = kwargs

        primary_index_cols = primary_df.index.names
        if not set(groupby_as_list).issubset(primary_index_cols):
            raise ValueError("All `groupby` columns need to be in the index of all dataframes.")

        self.primary_df = primary_df
        self.secondary_dfs = secondary_dfs
        if not isinstance(secondary_dfs, list):
            self.secondary_dfs = [secondary_dfs]
        self.groupby = groupby

    @property
    def primary_groupby(self) -> pd.core.groupby.DataFrameGroupBy:
        """The primary groupby object.

        This is the grouper created from the primary dataframe.
        """
        if not hasattr(self, "_primary_groupby"):
            self._primary_groupby = self.primary_df.groupby(level=self.groupby, **self._kwargs)
        return self._primary_groupby

    @property
    def secondary_groupbys(self) -> list[pd.core.groupby.DataFrameGroupBy]:
        """The secondary groupby objects.

        These are the groupers created from the secondary dataframes.
        """
        if not hasattr(self, "_secondary_groupbys"):
            self._secondary_groupbys = [df.groupby(level=self.groupby, **self._kwargs) for df in self.secondary_dfs]
        return self._secondary_groupbys

    def _get_secondary_vals(self, name: Union[str, tuple[str, ...]]) -> list[pd.DataFrame]:
        return [
            _get_group_with_empty_fallback(g, df, name) for g, df in zip(self.secondary_groupbys, self.secondary_dfs)
        ]

    def get_group(self, name: Union[str, tuple[str, ...]]) -> tuple[pd.DataFrame, ...]:
        """Get an individual group by name.

        Returns
        -------
        A tuple containing the groups from each dataframe.
        """
        return self.primary_groupby.get_group(name), *self._get_secondary_vals(name)

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
        return ((name, self.get_group(name)) for name, _ in self.primary_groupby)

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
    **kwargs: Unpack[dict[str, Any]],
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
    kwargs
        All further arguments will be passed to ``.groupby`` of all dataframes.

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
    return MultiGroupBy(primary_df, secondary_dfs, groupby, **kwargs)


class CustomOperation(NamedTuple):
    """Metadata for custom aggregations and transformations.

    Parameters
    ----------
    identifier
        The data identifier to select the relevant columns from the DataFrame.
        If `None`, the entire DataFrame is used.
        Otherwise, it needs to be a valid loc indexer for the DataFrame (`df.loc[:, identifier]`).
    function
        The function to apply.
        They will get the selected data as first argument.
        There expected return value depends on the context the CustomOperation is used in.
    column_name
        The name of the resulting column in the output dataframe.
        If a list of columns names is provided, the results of the custom function will be spread over multiple columns.
        For example, if the function returns a tuple of two values, the column names should be a tuple of two strings,
        if you want the results to be stored in two separate columns.
        If just a single string is provided, the results will be stored in a single column.

    """

    identifier: Union[Hashable, Sequence, str, None]
    function: Callable
    column_name: Union[str, tuple[str, ...], list[Union[str, tuple[str, ...]]]]

    @property
    def _TAG(self) -> str:  # noqa: N802
        return "CustomOperation"


class MissingDataColumnsError(ValueError):
    """Error raised when the columns specified in the transformations are not found in the DataFrame."""

    def __init__(self, missing_columns: Union[Hashable, Sequence, str]) -> None:
        self.missing_columns = missing_columns
        super().__init__(
            f"One of the transformations requires the following columns: '{missing_columns}'. "
            f"They are not found in the DataFrame."
        )


def _get_data_from_identifier(
    df: pd.DataFrame, identifier: Union[Hashable, Sequence, str, None], num_levels: Union[int, None] = 1
) -> pd.DataFrame:
    if identifier is None:
        return df
    try:
        data = df.loc[:, identifier]
    except KeyError as e:
        raise MissingDataColumnsError(identifier) from e
    if num_levels:
        data_num_levels = 1 if isinstance(data, pd.Series) else data.columns.nlevels
        if data_num_levels != num_levels:
            raise ValueError(f"Data selected by '{identifier}' must have {num_levels} level(s).")
    return data


def apply_transformations(  # noqa: C901, PLR0912
    df: pd.DataFrame,
    transformations: list[Union[tuple[str, Union[callable, list[callable]]], CustomOperation]],
    *,
    missing_columns: Literal["raise", "ignore", "warn"] = "warn",
) -> pd.DataFrame:
    """Apply a set of transformations to DataFrame.

    Compared to the default pandas ``df.transform`` method, this allows  more flexibility in selecting the data to
    apply the transformations to and in defining the transformations themselves.
    In particular, it allows to apply transformations that require multiple columns as input.

    Parameters
    ----------
    df
        The DataFrame containing the data to transform.
        This can have a single or multi-level column index.
        The identifiers provided for the transformations must be valid loc identifiers for the DataFrame.

    transformations

        A list specifying which transformation functions are to the df.
        They can be provided in two ways:

        1.  As a tuple in the format `(<identifier>, <function>)`,
            where `<identifier>` is a valid loc columns-indexer for the DataFrame, and `<function>` is the function
            (or a list of functions) to apply.
            When the identfier returns a sub-dataframe with multiple columns, then the function will get this entire
            subdataframe to operate on.
            However, we always expect the function to just return a single Series with the same number of rows as the
            dataframe.

        2.  As a named tuple of type :class:`CustomOperation` taking three arguments:
            `identifier`, `function`, and `column_name`.
            `identifier` is a valid loc identifier selecting one or more columns from the dataframe, `function` is the
            (custom) transformation function or list of functions to apply, and `column_name` is the name of the
            resulting column in the output dataframe.
            `column_name` provides the name of the resulting column in the output dataframe.
            This should either be a string or a tuple of strings, matching the "depth" of the `<identifier>` used in
            the normal transformations (if a combination is provided).
            This allows for more complex transformations that require multiple columns as input.
            We also support a special case, where the custom function returns a tuple of results (e.g. two Series).
            In this case, the `column_name` should be a list of strings or tuples of strings, where each string
            corresponds to one of the results returned by the function.
            Note, that your custom function MUST return a tuple in this case (not a list or other iterable).

    missing_columns
        How to handle missing columns specified in the transformations.

        - "raise": Raise a `MissingDataColumnsError`.
        - "ignore": Ignore the missing columns and continue with the remaining transformations.
        - "warn": Issue a warning and continue with the remaining transformations (default).

    Returns
    -------
    transformed_df
        Dataframe with the transformed values.
        The columns of the transformed DataFrame are multi-level and will have the form `(*idetifier, function_name)`

    Notes
    -----
    .. warning::
        When mixing custom operations with built-in aggregations, make sure that the number of levels in the identifiers
        of the normal aggregations and the number of levels in the `column_name` attribute of the custom aggregations
        are identical.
        Otherwise, they can not be combined.


    """
    transformation_results = []
    column_names = []
    for transformation in transformations:
        if getattr(transformation, "_TAG", None) == "CustomOperation":
            identifier = transformation.identifier
            functions = [transformation.function]
            if isinstance(transformation.column_name, list):
                col_names = transformation.column_name
            else:
                col_names = [transformation.column_name]
        else:
            identifier, functions = transformation
            col_names = []
            if not isinstance(functions, list):
                functions = [functions]
            for fct in functions:
                try:
                    fct_name = fct.__name__
                except AttributeError as e:
                    raise ValueError(
                        f"Transformation function {fct} for identifier {identifier} does not have a "
                        "`__name__`-Attribute. "
                        "Please use a named function or assign a name."
                    ) from e
                col_names.append((identifier, fct_name))

        for fct, col_name in zip(functions, col_names):
            try:
                data = _get_data_from_identifier(df, identifier, num_levels=None)
            except MissingDataColumnsError as e:
                if missing_columns == "raise":
                    raise
                if missing_columns == "warn":
                    warnings.warn(str(e), stacklevel=1)
                continue
            result = fct(data)
            if isinstance(result, tuple):
                assert len(result) == len(col_name)
                transformation_results.extend(result)
                column_names.extend(col_name)
            else:
                transformation_results.append(result)
                column_names.append(col_name)

    # combine results
    try:
        transformation_results = pd.concat(transformation_results, axis=1)
    except TypeError as e:
        raise ValueError(
            "The transformation results could not be concatenated. "
            "This is likely due to an unexpected return type of a custom function."
            "Please ensure that the return type is a pandas Series for all custom functions."
        ) from e
    if all(not isinstance(c, tuple) for c in column_names):
        # This should be a normal index not mutliindex
        transformation_results.columns = pd.Index(column_names)
        return transformation_results
    column_names = [col_name if isinstance(col_name, tuple) else (col_name,) for col_name in column_names]
    try:
        transformation_results.columns = pd.MultiIndex.from_tuples(column_names)
    except ValueError as e:
        raise ValueError(
            f"The expected number of column names {len(pd.MultiIndex.from_tuples(column_names))} "
            f"does not match with the actual number {transformation_results.shape[1]} of columns "
            "in the transformed DataFrame."
            "This is likely due to an unexpected return shape of a CustomOperation function."
        ) from e
    return transformation_results


def apply_aggregations(
    df: pd.DataFrame,
    aggregations: list[
        Union[
            tuple[Union[str, tuple[str, ...]], Union[Union[callable, str], list[Union[callable, str]]]], CustomOperation
        ]
    ],
    *,
    missing_columns: Literal["raise", "ignore", "warn"] = "warn",
) -> pd.Series:
    """Apply a set of aggregations to any Dataframe.

    Returns a Series with one entry per aggregation.
    Compared to the default pandas ``df.agg`` method, this allows more flexibility in selecting the data to apply the
    allows to apply aggregations, that require the data of multiple columns at once.

    Parameters
    ----------
    df
        The DataFrame containing the data to aggregate.
        Aggregations are applied on individual or multiple columns of this DataFrame.
        The identifier provided in the aggregations must be a valid loc identifier for the DataFrame.

    aggregations : list
        A list specifying which aggregation functions are to be applied for which metrics and data origins.
        There are two ways to define aggregations:

        1.  As a tuple in the format `(<identifier>, <aggregation>)`.
            In this case, the operation is performed based on exactly one column from the input df.
            Therefore, <identifier> can either be a string representing the name of the column to evaluate
            (for data with single-level columns),
            or a tuple of strings uniquely identifying the column to evaluate.
            `<aggregation>` is the function or the list of functions to apply.

        2.  As a named tuple of type `CustomOperation` taking three arguments:
            `identifier`, `function`, and `column_name`.
            `identifier` is a valid loc identifier selecting one or more columns from the dataframe, `function` is the
            (custom) aggregation function or list of functions to apply, and `column_name` is the name of the resulting
            column in the output dataframe.
            In case of a single-level output column, `column_name` is a string, whereas for multi-level output columns,
            it is a tuple of strings.
            This allows for more complex aggregations that require multiple columns as input,
    missing_columns
        How to handle missing columns specified in the aggregations.

        - "raise": Raise a `MissingDataColumnsError`.
        - "ignore": Ignore the missing columns and continue with the remaining aggregations.
        - "warn": Issue a warning and continue with the remaining aggregations (default).

    Returns
    -------
    aggregated_series
        A pandas series containing the aggregated values.
        The index of the series is defined by the identifiers of the aggregations and the names of the functions.
        The multiindex columns will have the form `(*idetifier, function_name)`

    Notes
    -----
    .. warning::
        When mixing custom operations with built-in aggregations, make sure that the number of levels in the identifiers
        of the normal aggregations and the number of levels in the `column_name` attribute of the custom aggregations
        are identical.
        Otherwise, they can not be combined.

    As implementation note, all the traditional aggregations will be directly handled by Pandas ``df.agg`` method.
    All the CustomOperations will be applied manually.
    At the end the results will be concatenated.

    """
    manual_aggregations, agg_aggregations = _collect_manual_and_agg_aggregations(aggregations)

    # apply built-in aggregations
    agg_aggregation_results = []
    for key, aggregation in agg_aggregations.items():
        try:
            aggregation_result = df.agg({key: aggregation})
            agg_aggregation_results.append(
                aggregation_result.stack(level=np.arange(df.columns.nlevels).tolist(), future_stack=True)
            )
        except KeyError as e:
            if missing_columns == "raise":
                raise MissingDataColumnsError(key) from e
            if missing_columns == "warn":
                warnings.warn(str(MissingDataColumnsError(key)), UserWarning, stacklevel=1)
            continue
    agg_aggregation_results = pd.concat(agg_aggregation_results) if agg_aggregation_results else pd.Series()

    manual_aggregation_results = _apply_manual_aggregations(df, manual_aggregations, missing_columns)

    # if only one type of aggregation was applied, return the result directly
    if manual_aggregations and not agg_aggregations:
        return manual_aggregation_results
    if agg_aggregations and not manual_aggregations:
        return agg_aggregation_results

    # otherwise, concatenate the results
    try:
        _check_number_of_index_levels([agg_aggregation_results, manual_aggregation_results])
    except ValueError as e:
        raise ValueError(
            "The aggregation results from automatic and custom aggregation could not be concatenated. "
            "This is likely caused by an inconsistent number index levels in them."
        ) from e
    aggregation_results = pd.concat([agg_aggregation_results, manual_aggregation_results])

    return aggregation_results


def _collect_manual_and_agg_aggregations(
    aggregations: list[
        Union[
            tuple[Union[str, tuple[str, ...]], Union[Union[callable, str], list[Union[callable, str]]]], CustomOperation
        ]
    ],
) -> tuple[list[CustomOperation], dict[tuple[str, str], list[Union[str, Callable]]]]:
    manual_aggregations = []
    agg_aggregations = {}
    for agg in aggregations:
        if getattr(agg, "_TAG", None) == "CustomOperation":
            manual_aggregations.append(agg)
        else:
            key, aggregation = agg
            if not isinstance(aggregation, list):
                aggregation = [aggregation]
            wrapped_aggregation = []
            for fct in aggregation:
                if isinstance(fct, str):
                    # skip special case string-functions (e.g. "mean")
                    wrapped_aggregation.append(fct)
                else:
                    # wrap function to prevent unexpected behavior of pd.DataFrame.agg
                    # otherwise, data is internally passed to apply element-wise instead of as whole series
                    # for user-defined functions: https://github.com/pandas-dev/pandas/issues/41768
                    wrapped_aggregation.append(_allow_only_series(fct))
            # agg function only accepts strings as identifiers for one-level columns
            if isinstance(key, tuple) and len(key) == 1:
                key = key[0]
            if not isinstance(key, (tuple, str)) or not all(isinstance(k, str) for k in key):
                raise ValueError(
                    f"The key {key} has an invalid type. "
                    "It must either be a valid column name or a tuple of column names."
                )
            agg_aggregations.setdefault(key, []).extend(wrapped_aggregation)
    return manual_aggregations, agg_aggregations


def _allow_only_series(func: callable) -> callable:
    # if data are passed to apply element-wise,
    # throw an error to ensure that they are processed as whole series
    @wraps(func)
    def wrapper(x: pd.Series) -> Any:
        if not isinstance(x, (pd.Series, pd.DataFrame)):
            raise TypeError("Only Series allowed as input.")
        return func(x)

    return wrapper


def _construct_index_from_col_name(col_name: Union[tuple[str, ...], str]) -> pd.Index:
    if isinstance(col_name, tuple):
        return pd.MultiIndex.from_tuples([col_name])
    return pd.Index([col_name])


def _apply_manual_aggregations(  # noqa: C901
    df: pd.DataFrame, manual_aggregations: list[CustomOperation], missing_columns: Literal["raise", "ignore", "warn"]
) -> pd.Series:
    # apply manual aggregations
    manual_aggregation_results = []
    for agg in manual_aggregations:
        agg_function = agg.function

        try:
            data = _get_data_from_identifier(df, agg.identifier, num_levels=None)
        except MissingDataColumnsError as e:
            if missing_columns == "raise":
                raise
            if missing_columns == "warn":
                warnings.warn(str(e), UserWarning, stacklevel=1)
            continue

        result = agg_function(data)
        if not agg.column_name:
            raise ValueError(
                "Custom aggregations always need to specify a column name that includes the metric."
                "E.g. ('icc', 'speed_error') for the ICC of the speed error."
            )
        if isinstance(agg.column_name, list):
            # We need to spread the results over multiple columns
            result = result if isinstance(result, tuple) else (result,)
            if not len(agg.column_name) == len(result):
                raise ValueError(
                    "The number of column names provided does not match the number of results returned by the function."
                )
            for col_name, res in zip(agg.column_name, result):
                manual_aggregation_results.append(pd.Series([res], index=_construct_index_from_col_name(col_name)))
        else:
            manual_aggregation_results.append(
                pd.Series([result], index=_construct_index_from_col_name(agg.column_name))
            )
    if len(manual_aggregation_results) == 0:
        return pd.Series()
    try:
        _check_number_of_index_levels(manual_aggregation_results)
    except ValueError as e:
        raise ValueError(
            "Error in concatenating manual aggregation results. "
            "Please ensure that the `col_names` attribute has the same number of elements "
            "across all custom aggregations"
        ) from e
    return pd.concat(manual_aggregation_results)


def _check_number_of_index_levels(agg_results: list[Union[pd.Series, pd.DataFrame]]) -> None:
    n_levels = [result.index.nlevels for result in agg_results]
    if len(set(n_levels)) > 1:
        raise ValueError(
            "Number of index levels in results is not consistent. "
            "Please ensure that all aggregation results have the same number of index levels."
        )


def cut_into_overlapping_bins(
    df: pd.DataFrame, column: str, interval_dict: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    """
    Cut a DataFrame column into potentially overlapping bins.

    .. warning:: Overlapping areas of the intervals will be duplicated in the output DataFrame.
       This could lead to large performance slowdowns.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to bin
    column : str
        The name of the column to bin
    interval_dict : dict
        Dictionary where keys are bin labels and values are (min, max) tuples
        defining the intervals, e.g. {'all': (-np.inf, np.inf), '10+': (10, np.inf)}

    Returns
    -------
    pandas.DataFrame
        DataFrame with the same columns as the orignal df and an addtion 'bin' (categorical bin labels) columns,
        with potentially duplicated rows for values that fall into multiple bins
    """
    # Create a list to store rows for the new DataFrame
    col_of_interest = df[column]
    groups = {key: df[(col_of_interest > val[0]) & (col_of_interest <= val[1])] for key, val in interval_dict.items()}

    return (
        pd.concat(groups, names=["bin", *df.index.names])
        .reset_index("bin")
        .astype({"bin": pd.CategoricalDtype(list(interval_dict.keys()), ordered=True)})
    )


def multilevel_groupby_apply_merge(
    df: pd.DataFrame, groupbys: list[tuple[Union[str, list[str]], Callable]], **apply_kwargs: Any
) -> pd.DataFrame:
    """Apply multiple groupby operations and merge the results.

    This function allows to apply multiple groupby operations to a DataFrame and merge the results into a
    single DataFrame.
    The groupby operations are defined by a dictionary, where the keys are the groupby columns and the values are the
    aggregation functions to apply.
    All groupby results must have the same shape to be able to merge them.

    Parameters
    ----------
    df
        The DataFrame to apply the groupby operations on.
    groupbys
        A dictionary where keys are the groupby columns and values are the aggregation functions to apply.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the results of the groupby operations.
    """
    results = [df.groupby(key).apply(func, include_groups=False, **apply_kwargs) for key, func in groupbys]
    return pd.concat(results, axis=1) if results else pd.DataFrame()


__all__ = [
    "CustomOperation",
    "MultiGroupBy",
    "MultiGroupByPrimaryDfEmptyError",
    "apply_aggregations",
    "apply_transformations",
    "create_multi_groupby",
    "cut_into_overlapping_bins",
    "multilevel_groupby_apply_merge",
]
