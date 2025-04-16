import warnings
from typing import Final, Literal, Union

import numpy as np
import pandas as pd
from pingouin import intraclass_corr

from mobgap.utils.df_operations import CustomOperation, _get_data_from_identifier


def _handle_zero_division(
    divisor: Union[pd.Series, pd.DataFrame],
    zero_division_hint: Union[Literal["warn", "raise"], float],
    caller_fct_name: str,
) -> None:
    if (divisor == 0).any():
        if zero_division_hint not in ["warn", "raise", np.nan]:
            raise ValueError('"zero_division" must be set to "warn", "raise" or `np.nan`!')
        if zero_division_hint == "raise":
            raise ZeroDivisionError(f"Zero division occurred in {caller_fct_name} because divisor contains zeroes.")
        if zero_division_hint == "warn":
            warnings.warn(
                f"Zero division occurred in {caller_fct_name} because divisor contains zeroes. "
                "Affected error metrics are set to NaN.",
                UserWarning,
                stacklevel=2,
            )


def error(df: pd.DataFrame, reference_col_name: str = "reference", detected_col_name: str = "detected") -> pd.Series:
    """
    Calculate the error between the detected and reference values.

    Parameters
    ----------
    df
        The DataFrame containing the reference and detected values.
    reference_col_name
        The identifier of the column containing the reference values.
    detected_col_name
        The identifier of the column containing the detected values.

    Returns
    -------
    error
        The error between the detected and reference values in the form `detected` - `reference`
    """
    ref, det = _get_data_from_identifier(df, reference_col_name), _get_data_from_identifier(df, detected_col_name)
    return det - ref


def rel_error(
    df: pd.DataFrame,
    reference_col_name: str = "reference",
    detected_col_name: str = "detected",
    zero_division_hint: Union[Literal["warn", "raise"], float] = "warn",
) -> pd.Series:
    """
    Calculate the relative error between the detected and reference values.

    Parameters
    ----------
    df
        The DataFrame containing the reference and detected values.
    reference_col_name
        The identifier of the column containing the reference values.
    detected_col_name
        The identifier of the column containing the detected values.
    zero_division_hint
        How to handle zero division errors. Can be one of "warn" (warning is given, respective values are set to NaN),
        "raise" (error is raised), or "np.nan" (respective values are silently set to NaN).

    Returns
    -------
    rel_error
        The relative error between the detected and reference values
        in the form (`detected` - `reference`) / `reference`.
    """
    ref, det = (
        _get_data_from_identifier(df, reference_col_name),
        _get_data_from_identifier(df, detected_col_name),
    )
    # inform about zero division if it occurs
    _handle_zero_division(ref, zero_division_hint, "rel_error")
    result = (det - ref) / ref
    with pd.option_context("future.no_silent_downcasting", True):
        result = result.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)
    return result


def abs_error(
    df: pd.DataFrame, reference_col_name: str = "reference", detected_col_name: str = "detected"
) -> pd.Series:
    """
    Calculate the absolute error between the detected and reference values.

    Parameters
    ----------
    df
        The DataFrame containing the reference and detected values.
    reference_col_name
        The identifier of the column containing the reference values.
    detected_col_name
        The identifier of the column containing the detected values.

    Returns
    -------
    abs_error
        The absolute error between the detected and reference values in the form `abs(detected - reference)`.
    """
    ref, det = _get_data_from_identifier(df, reference_col_name), _get_data_from_identifier(df, detected_col_name)
    return abs(det - ref)


def abs_rel_error(
    df: pd.DataFrame,
    reference_col_name: str = "reference",
    detected_col_name: str = "detected",
    zero_division_hint: Union[Literal["warn", "raise"], float] = "warn",
) -> pd.Series:
    """
    Calculate the absolute relative error between the detected and reference values.

    Parameters
    ----------
    df
        The DataFrame containing the reference and detected values.
    reference_col_name
        The identifier of the column containing the reference values.
    detected_col_name
        The identifier of the column containing the detected values.
    zero_division_hint
        How to handle zero division errors. Can be one of "warn" (warning is given, respective values are set to NaN),
        "raise" (error is raised), or "np.nan" (respective values are silently set to NaN).

    Returns
    -------
    abs_rel_error
        The absolute relative error between the detected and reference values
        in the form `abs((detected - reference) / reference)`.
    """
    ref, det = (
        _get_data_from_identifier(df, reference_col_name),
        _get_data_from_identifier(df, detected_col_name),
    )
    # inform about zero division if it occurs
    _handle_zero_division(ref, zero_division_hint, "abs_rel_error")
    result = abs((det - ref) / ref)
    with pd.option_context("future.no_silent_downcasting", True):
        result = result.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)
    return result


class ErrorTransformFuncs:
    """Typical row by row error functions.

    All functions expect a dataframe `df` as input that contains a column called ``reference`` and a column called
    ``detected`` (per default).
    The name of the columns can be changed using the ``reference_col_name`` and ``detected_col_name`` parameters.

    The functions return a series with the same index as the input dataframe.

    Attributes
    ----------
    error
        Simple error defined as `detected - reference`.
    rel_error
        Relative error defined as `(detected - reference) / reference`.
        Supports additional ``zero_division_hint`` parameter to control what happens in case `reference` is 0
    abs_error
        Absolute error defined as `abs(detected - reference)`.
    abs_rel_error
        Absolute relative error defined as `abs((detected - reference) / reference)`.
        Supports additional ``zero_division_hint`` parameter to control what happens in case `reference` is 0

    """

    error: Final = error
    rel_error: Final = rel_error
    abs_error: Final = abs_error
    abs_rel_error: Final = abs_rel_error


def get_default_error_transformations() -> list[tuple[str, list[callable]]]:
    """
    Get all default error metrics used in Mobilise-D.

    This list can directly be passed to ~func:`~mobgap.utils.df_operations.apply_transformations` as the
    `transformations` parameter to calculate the desired metrics.
    """
    metrics = [
        "cadence_spm",
        "duration_s",
        "n_steps",
        "n_strides",
        "n_turns",
        "stride_duration_s",
        "stride_length_m",
        "walking_speed_mps",
    ]
    default_errors = [error, rel_error, abs_error, abs_rel_error]
    error_metrics = [*((m, default_errors) for m in metrics)]
    return error_metrics


def icc(
    df: pd.DataFrame,
    reference_col_name: str = "reference",
    detected_col_name: str = "detected",
    *,
    icc_type: str = "icc2",
    nan_policy: Literal["raise", "omit"] = "raise",
) -> tuple[float, tuple[float, float]]:
    """
    Calculate the intraclass correlation coefficient (ICC) for the detected and reference values.

    Parameters
    ----------
    df
        The DataFrame containing the reference and detected values.
    reference_col_name
        The identifier of the column containing the reference values.
    detected_col_name
        The identifier of the column containing the detected values.
    icc_type
        The type of the ICC. Can be one of "icc1", "icc2", "icc3", "icc1k", "icc2k", "icc3k".
        See the documentation of the `pingouin.intraclass_corr` function for more information.
        Default is "icc2", often also referred to as ICC(2,1).
    nan_policy
        How to handle NaN values. Can be one of "raise" (error is raised), or "omit" (NaN values are ignored).
        Default is "raise".

    Notes
    -----
    Note, that in case of ICC2, the confidence interval is reported as [np.nan, np.nan] if the ICC is 1 or 0 (aka
    perfect agreement or disagreement) as the confidence interval is not defined in this case.
    Other implementations might return [1, 1] in this case.

    Returns
    -------
    icc, ci95
        A tuple containing the intraclass correlation coefficient (ICC) as first item
        and the lower and upper bound of its 95% confidence interval (CI95%) as second item.

    """
    df = _get_data_from_identifier(df, [reference_col_name, detected_col_name], num_levels=1)
    df = (
        df.astype(float)
        .reset_index(drop=True)
        .rename_axis("targets", axis=0)
        .rename_axis("rater", axis=1)
        .stack()
        .rename("value")
        .reset_index()
    )
    icc, ci95 = (
        intraclass_corr(data=df, targets="targets", raters="rater", ratings="value", nan_policy=nan_policy)
        .set_index("Type")
        .loc[icc_type.upper(), ["ICC", "CI95%"]]
    )
    return float(icc), tuple(float(v) for v in ci95)


def quantiles(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> tuple[float, float]:
    """Calculate the quantiles of a measure.

    Parameters
    ----------
    series
        The Series containing the data column of interest.
    lower
        The lower quantile to calculate.
    upper
        The upper quantile to calculate.

    Returns
    -------
    quantiles
        The lower and upper quantiles as a tuple.
    """
    return tuple(series.quantile([lower, upper]).to_list())


def loa(series: pd.Series, agreement: float = 1.96) -> tuple[float, float]:
    """Calculate the limits of agreement of a measure.

    Parameters
    ----------
    series
        The Series containing the data column of interest.
    agreement
        The agreement level for the limits of agreement.

    Returns
    -------
    loa
        The lower and upper limits of agreement as a tuple.
    """
    mean = series.mean()
    std = series.std()
    return float(mean - std * agreement), float(mean + std * agreement)


def conf_intervals(series: pd.Series, z_score: float = 1.96) -> tuple[float, float]:
    """Calculate the confidence intervals of a measure.

    Parameters
    ----------
    series
        The Series containing the data column of interest.
    z_score
        The agreement level for the limits of agreement.

    Returns
    -------
    conf_intervals
        The lower and upper confidence intervals as a tuple.
    """
    mean = series.mean()
    std = series.std()
    se = std / np.sqrt(len(series))
    return float(mean - se * z_score), float(mean + se * z_score)


def n_datapoints(df: pd.DataFrame) -> int:
    return len(df)


class CustomErrorAggregations:
    """Custom aggregation functions that might be useful in addition to the once provided by pandas (e.g. mean/std).

    The functions are designed to work in combination with the :func:`~mobgap.utils.df_operations.apply_aggregations`.


    Attributes
    ----------
    icc
        Calculate the intraclass correlation coefficient (ICC) for the detected and reference values.
        This requires a dataframe with multiple columns as input.
        Returns two values: the ICC and the CI95% values as tuple.
    quantiles
        Calculate the quantiles of a measure (95% by default).
        This requires a series as input.
        Returns a tuple with the lower and upper quantiles.
    loa
        Calculate the limits of agreement of a measure.
        This requires a series as input.
        Returns a tuple with the lower and upper limits of agreement.
    conf_intervals
        Calculate the confidence intervals of a measure.
        This requires a series as input.
        Returns a tuple with the lower and upper confidence intervals.
        Calculates the 95% confidence intervals (z-score 1.96) by default.
    n_datapoints
        Calculate the number of datapoints in a dataframe.

    """

    icc = icc
    quantiles = quantiles
    loa = loa
    conf_intervals = conf_intervals
    n_datapoints = n_datapoints


def get_default_error_aggregations() -> list[
    Union[tuple[tuple[str, ...], Union[list[Union[callable, str]], callable, str]], CustomOperation]
]:
    """Return a list containing all important error aggregations utilized in Mobilise-D.

    This list can directly be passed to ~func:`~mobgap.utils.df_operations.apply_aggregations` as the `aggregations`
    parameter to calculate the desired metrics.
    """
    metrics = [
        "cadence_spm",
        "duration_s",
        "n_steps",
        "n_strides",
        "n_turns",
        "stride_duration_s",
        "stride_length_m",
        "walking_speed_mps",
    ]

    default_agg = [
        *(
            ((m, o), ["mean", quantiles])
            for m in metrics
            for o in ["detected", "reference", "abs_error", "abs_rel_error"]
        ),
        *(((m, o), ["mean", loa]) for m in metrics for o in ["error", "rel_error"]),
        *[CustomOperation(identifier=m, function=icc, column_name=("icc", m, "all")) for m in metrics],
        CustomOperation(identifier=None, function=n_datapoints, column_name=("n_datapoints", "all", "all")),
    ]

    return default_agg
