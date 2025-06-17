"""Tools to format pandas tables for documentation and publication."""

import operator
from collections.abc import Hashable
from typing import Any, Callable, NamedTuple, Optional, Union

import pandas as pd
import pingouin as pg
from pandas.io.formats.style import Styler


class ValueWithMetadata:
    """A base class to represent a value with associated metadata.

    This can be used as a value within a pandas DataFrame, allowing for custom formatting.
    For this create a subclass that implements the `__str__` and `_repr_html_` methods.
    The main value provided is still used for comparisons, so you can use this class while allowing for sorting and
    filtering.
    """

    def __init__(self, value: float, metadata: Optional[dict[str, Any]] = None, precision: int = 2) -> None:
        """Initialize the ValueWithMetadata object.

        Parameters
        ----------
        value
            The value to be stored.
        metadata
            Optional metadata associated with the value.
        precision
            The number of decimal places to use for formatting the value.
        """
        self.value = value
        self.metadata = metadata or {}
        self.precision = precision

    def _compare(self, other: Any, comparison: Callable[[float, Any], bool]) -> bool:
        if not isinstance(other, ValueWithMetadata):
            return comparison(self.value, other)
        return comparison(self.value, other.value)

    def __str__(self) -> str:  # noqa: D105
        raise NotImplementedError

    def _repr_html_(self) -> str:
        """HTML representation of the value.

        Note that this will only show up, when applying the `html_styler` method to a DataFrame.
        """
        raise NotImplementedError

    def __lt__(self, other: Any) -> bool:  # noqa: D105
        return self._compare(other, operator.lt)

    def __le__(self, other: Any) -> bool:  # noqa: D105
        return self._compare(other, operator.le)

    def __gt__(self, other: Any) -> bool:  # noqa: D105
        return self._compare(other, operator.gt)

    def __ge__(self, other: Any) -> bool:  # noqa: D105
        return self._compare(other, operator.ge)

    @classmethod
    def style_html(cls, value: Any) -> Any:
        """Return a string representation of the value for styling purposes."""
        if not isinstance(value, cls):
            return value

        return value._repr_html_()

    @classmethod
    def html_styler(cls, st: Styler) -> Styler:
        """Apply HTML styling to the DataFrame using the custom HTML representation."""
        return st.format(cls.style_html)


class CustomFormattedValueWithMetadata(ValueWithMetadata):
    """A custom object to represent a value with a range and stats results.

    In the context of comparisons, it acts like it's value, but it can also be used to display the value and range.
    """

    def _create_p_val_format(self) -> Optional[str]:
        stats_metadata = self.metadata.get("stats_metadata") or {}
        if (p_val := stats_metadata.get("p")) is None:
            return None
        if p_val < 0.01:
            return "**"
        if p_val < 0.05:
            return "*"
        return None

    def __str__(self) -> str:  # noqa: D105
        postfix = self._create_p_val_format() or ""
        err_range = self.metadata.get("range")
        if err_range is None:
            return f"{self.value:.{self.precision}f}{postfix}"
        return (
            f"{self.value:.{self.precision}f} [{err_range[0]:.{self.precision}f}, "
            f"{err_range[1]:.{self.precision}f}]{postfix}"
        )

    def _repr_html_(self) -> str:
        """Return a string representation of the value with metadata for HTML rendering."""
        postfix = f"<sup>{p}</sup>" if (p := self._create_p_val_format()) else ""
        err_range = self.metadata.get("range")
        if err_range is None:
            return f"<span>{self.value:.{self.precision}f}</span>{postfix}"
        return (
            f"<span>{self.value:.{self.precision}f} "
            f"[{err_range[0]:.{self.precision}f}, {err_range[1]:.{self.precision}f}]</span>{postfix}"
        )


def value_with_metadata(
    df: pd.DataFrame,
    value_col: Hashable,
    other_columns: dict[str, Hashable],
    precision: int = 2,
    base_class: type[ValueWithMetadata] = CustomFormattedValueWithMetadata,
) -> pd.Series:
    """Combine a value column (float) and additional metadata columns into one column.

    Note, that the return value is not a string, but a custom object that has the expected string representation.
    This means that if you apply this to a pandas dataframe, you can still perform regular comparisons with the values.
    The comparisons are based on the value only and not the range.
    Don't overuse this "magic trick".
    We are just using it to still apply styles based on the value in the final dataframe.

    Parameters
    ----------
    df
        The DataFrame containing the columns.
    value_col
        The name of the column containing the value.
        This is important, as the value is used for comparisons/sorting.
    other_columns
        A dictionary mapping keys to column names that contain the further metadata.
    precision
        numbers of decimal places to use for all values during formatting.
    base_class
        The base class to use for the custom object.
        This can be used to create custom formatters that make use of the metadata.

    """

    def get_value(row: pd.Series, col: Hashable) -> Any:
        """Get the value from the row, handling NaN values."""
        value = row.get(col)
        if pd.isna(value):
            return None
        return value

    def transform_values(row: pd.Series) -> base_class:
        """Extract values from the row based on the provided columns."""
        return base_class(
            value=row[value_col],
            metadata={key: get_value(row, value) for key, value in other_columns.items()},
            precision=precision,
        )

    return df.apply(
        transform_values,
        axis=1,
    )


class FormatTransformer:
    """Formatting functions that can be applied to a DataFrame using :func:`~mobgap.utils.df_operations.apply_transformations`.

    These functions can be applied to individual columns or to multiple columns combined
    by using :class:`~mobgap.utils.df_operations.CustomOperation`.
    If you want to perform styling per value, usually the built-in pandas styling functions are more appropriate.

    Attributes
    ----------
    value_with_metadata
        Combine a value column (float) and other metadata columns into a string. By default this supports ranges and
        statistics metadata.
    """  # noqa: E501

    value_with_metadata = value_with_metadata


def pairwise_tests(
    df: pd.DataFrame,
    value_col: str,
    between: str,
    reference_group_key: str,
) -> pd.Series:
    # We need to force a consistent order where the reference group is always the first.
    groups = set(df[between].unique())
    if reference_group_key not in groups:
        # If we don't have the reference group, we can't perform the tests.
        # This might happen for algorithms that do not have a reference algorithm.
        return pd.Series(pd.NA, index=groups)
    order = [reference_group_key, *sorted(groups - {reference_group_key})]
    df = df.assign(**{between: pd.Categorical(df[between], categories=order, ordered=True)})
    result = pg.pairwise_tests(data=df, dv=value_col, between=between)
    assert result["Paired"].eq(False).all(), "Expected unpaired tests"
    assert reference_group_key not in result["B"]
    return (
        result.query("A == @reference_group_key")
        .copy()
        .rename(columns={"p-unc": "p", "B": "version"})[["version", "T", "p"]]
        .set_index("version")
        .reindex(order)
        .apply(lambda row: row.to_dict(), axis=1)
    )


class StatsFunctions:
    """A collection of statistical functions that can be applied to a DataFrame.

    They are very specifically designed to work in the context of the mobgap revalidaition.
    It is very likely that your data shapes will not work with these functions.

    """

    pairwise_tests = pairwise_tests


def best_in_group_styler(
    groupby: Union[Hashable, list[Hashable]], columns: dict[Hashable, bool], style: str = "font-weight: bold"
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Style the best element in a column, after grouping by the index or another column.

    Parameters
    ----------
    groupby
        A valid groupby argument for :func:`~pandas.DataFrame.groupby`.
    columns
        A dictionary with the column names as keys and a boolean value if the best element is the largest or smallest.
        True for "larger is better", False for "smaller is better".
    style
        The CSS style to apply to the best element.
    """

    def style_best_in_group_column(data: pd.DataFrame) -> pd.DataFrame:
        styled = pd.DataFrame("", index=data.index, columns=data.columns)
        cols = list(columns.keys())
        grouped = data.groupby(groupby)[cols].agg(["idxmin", "idxmax"])
        for col, is_larger_better in columns.items():
            selector = (col,) if not isinstance(col, tuple) else col
            best_idx = grouped[(*selector, "idxmax" if is_larger_better else "idxmin")]
            styled.loc[best_idx, col] = style
        return styled

    return style_best_in_group_column


def border_after_group_styler(
    groupby: Union[Hashable, list[Hashable]], style: str = "border-bottom: 2px solid black"
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Add a border after each group in a DataFrame.

    Parameters
    ----------
    groupby
        A valid groupby argument for :func:`~pandas.DataFrame.groupby`.
    style
        The CSS style to apply to the border.
    """

    def style_border_after_group(data: pd.DataFrame) -> pd.DataFrame:
        styled = pd.DataFrame("", index=data.index, columns=data.columns)
        last_of_each_group = data.groupby(groupby).apply(lambda x: x.index[-1]).to_list()
        styled.loc[last_of_each_group, :] = style

        return styled

    return style_border_after_group


def compare_to_threshold_styler(
    thresholds: dict[Hashable, float],
    higher_is_pass: dict[Hashable, float],
    pass_style: str = "background-color: lightgreen",
    fail_style: str = "background-color: lightcoral",
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Style the elements in a DataFrame that are above or below a threshold.

    Parameters
    ----------
    thresholds
        A dictionary with the column names as keys and the threshold values as values.
    higher_is_pass
        A dictionary with the column names as keys and the threshold values as values.
    pass_style
        The CSS style to apply to the elements that pass the threshold.
    fail_style
        The CSS style to apply to the elements that fail the threshold.
    """

    def style_compare_to_threshold(data: pd.DataFrame) -> pd.DataFrame:
        styled = pd.DataFrame("", index=data.index, columns=data.columns)
        for col, threshold in thresholds.items():
            mask = data[col] > threshold if higher_is_pass[col] else data[col] < threshold
            styled.loc[mask, col] = pass_style
            styled.loc[~mask, col] = fail_style
        return styled

    return style_compare_to_threshold


class RevalidationInfo(NamedTuple):
    """Information required to determine the validity of a results in the context of the revalidation.

    Parameters
    ----------
    threshold
        Quality threshold of a specific error metric.
    higher_is_better
        Whether a higher value is better for the error metric.

    """

    threshold: Optional[float]
    higher_is_better: Optional[bool]


def revalidation_table_styles(
    st: Styler,
    thresholds: dict[Hashable, RevalidationInfo],
    groupby: Union[Hashable, list[Hashable]],
) -> Styler:
    """Apply styles to a DataFrame appropriate for the revalidation.

    To use this, set up a dictionary with the column names as keys and the comparison information as values.
    then you can run `df.style.pipe(revalidation_table_styles, thresholds, groupby=<groupby>)` to apply the styles.

    This applies ``best_in_group_styler``, ``compare_to_threshold_styler``, and ``border_after_group_styler`` with the
    default styles.

    Parameters
    ----------
    st
        The Styler object to apply the styles to.
    thresholds
        A dictionary with the column names as keys and the comparison information as values.
    groupby
        A valid groupby argument for :func:`~pandas.DataFrame.groupby`.
        This is used by ``best_in_group_styler`` and ``border_after_group_styler``.
    """
    higher_is_better = {
        col: info.higher_is_better for col, info in thresholds.items() if info.higher_is_better is not None
    }
    thresholds = {col: info.threshold for col, info in thresholds.items() if info.threshold is not None}

    return (
        st.pipe(ValueWithMetadata.html_styler)
        .apply(best_in_group_styler(groupby=groupby, columns=higher_is_better), axis=None)
        .apply(compare_to_threshold_styler(thresholds, higher_is_better), axis=None)
        .apply(border_after_group_styler(groupby), axis=None)
        .set_table_attributes('class="dataframe"')
    )


__all__ = [
    "CustomFormattedValueWithMetadata",
    "FormatTransformer",
    "RevalidationInfo",
    "StatsFunctions",
    "ValueWithMetadata",
    "best_in_group_styler",
    "border_after_group_styler",
    "compare_to_threshold_styler",
    "revalidation_table_styles",
]
