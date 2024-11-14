"""Tools to format pandas tables for documentation and publication."""

import pandas as pd


def value_with_range(df: pd.DataFrame, value_col: str, range_col: str, precision: int = 2) -> pd.Series:
    """Combine a value column (float) and a range column tuple(float, float) into one string.

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
        lambda x: f"{x[value_col]:.{precision}f} [{x[range_col][0]:.{precision}f}, {x[range_col][1]:.{precision}f}]",
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
