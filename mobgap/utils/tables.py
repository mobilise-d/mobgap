"""Tools to format pandas tables for documentation and publication."""

import operator
from collections.abc import Hashable
from typing import Any, Callable, NamedTuple, Optional, Union

import pandas as pd
from pandas.io.formats.style import Styler


class ValueWithRange:
    """A custom object to represent a value with a range.

    In the context of comparisons, it acts like it's value, but it can also be used to display the value and range.
    """

    def __init__(self, value: float, err_range: tuple[float, float], precision: int) -> None:
        self.value = value
        self.err_range = err_range
        self.precision = precision

    def _compare(self, other: Any, comparison: Callable[[float, Any], bool]) -> bool:
        if not isinstance(other, ValueWithRange):
            return comparison(self.value, other)
        return comparison(self.value, other.value)

    def __lt__(self, other: Any) -> bool:  # noqa: D105
        return self._compare(other, operator.lt)

    def __le__(self, other: Any) -> bool:  # noqa: D105
        return self._compare(other, operator.le)

    def __gt__(self, other: Any) -> bool:  # noqa: D105
        return self._compare(other, operator.gt)

    def __ge__(self, other: Any) -> bool:  # noqa: D105
        return self._compare(other, operator.ge)

    def __str__(self):  # noqa: ANN204, D105
        return (
            f"{self.value:.{self.precision}f} [{self.err_range[0]:.{self.precision}f}, "
            f"{self.err_range[1]:.{self.precision}f}]"
        )


def value_with_range(df: pd.DataFrame, value_col: str, range_col: str, precision: int = 2) -> pd.Series:
    """Combine a value column (float) and a range column tuple(float, float) into one column.

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
    range_col
        The name of the column containing the range.
    precision
        The precision to use for the value and range.

    """
    return df.apply(
        lambda row: ValueWithRange(row[value_col], row[range_col], precision),
        axis=1,
    )


class FormatTransformer:
    """Formatting functions that can be applied to a DataFrame using :func:`~mobgap.utils.df_operations.apply_transformations`.

    These functions can be applied to individual columns or to multiple columns combined
    by using :class:`~mobgap.utils.df_operations.CustomOperation`.
    If you want to perform styling per value, usually the built-in pandas styling functions are more appropriate.

    Attributes
    ----------
    value_with_range
        Combine a value column (float) and a range column tuple(float, float) into one string.
    """  # noqa: E501

    value_with_range = value_with_range


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
    st: Styler, thresholds: dict[Hashable, RevalidationInfo], groupby: Union[Hashable, list[Hashable]]
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
        st.apply(best_in_group_styler(groupby=groupby, columns=higher_is_better), axis=None)
        .apply(compare_to_threshold_styler(thresholds, higher_is_better), axis=None)
        .apply(border_after_group_styler(groupby), axis=None)
        .set_table_attributes('class="dataframe"')
    )


__all__ = [
    "FormatTransformer",
    "best_in_group_styler",
    "border_after_group_styler",
    "compare_to_threshold_styler",
    "revalidation_table_styles",
    "ValueWithRange",
    "RevalidationInfo",
]
