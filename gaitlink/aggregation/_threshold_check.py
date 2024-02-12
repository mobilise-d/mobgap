from collections.abc import Sequence
from importlib.resources import files

import numpy as np
import pandas as pd
from typing_extensions import Literal


def get_mobilised_dmo_thresholds() -> pd.DataFrame:
    """

    Read mobilised DMO (Dynamic Movement Orthosis) thresholds from a CSV file.

    Returns
    -------
        pd.DataFrame: Mobilised DMO thresholds with a multi-level index.

    """
    with (
        files("gaitlink") / "aggregation/_dmo_thresholds/official_mobilised_dmo_thresholds.csv"
    ).open() as threshold_data:
        thresholds = pd.read_csv(threshold_data, header=0, index_col=[0, 1])
    thresholds = thresholds.fillna(method="bfill").drop("ALL", level=1)
    thresholds.columns = pd.MultiIndex.from_tuples(
        [c.rsplit("_", 1) for c in thresholds.columns], names=["condition", "threshold_type"]
    )
    return thresholds


def _max_allowable_stride_length(height_m: float) -> float:
    """

    Calculate the maximum allowable stride length based on the participant's height.

    Args:
        height_m (float): Participant's height in meters.

    Returns
    -------
        float: Maximum allowable stride length.
    """
    leg_length = 0.53 * height_m
    froude_number = 1
    v_max = np.sqrt(froude_number * 9.81 * leg_length)
    h = 0.038 * v_max**2
    max_sl = 2 * 2 * np.sqrt(2 * leg_length * h - h**2)
    return max_sl


def apply_thresholds(
    input_data: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    cohort: str,
    height_m: float,
    condition: Literal["free_living", "laboratory"] = "free_living",
    check_against: Sequence[Literal["literature", "global"]] = ("literature", "global"),
) -> pd.DataFrame:
    """
    Apply DMO thresholds to input data and return a DataFrame check whether each data point is within the threshold.

    Parameters
    ----------
    input_data : pd.DataFrame
        Input data containing columns 'cadence_spm', 'walking_speed_mps', 'stride_length_m', 'stride_duration_s'.
    thresholds : pd.DataFrame
        DMO thresholds data.
    cohort : str
        Cohort identifier for filtering thresholds.
    height_m : float
        Participant's height in meters.
    condition : Literal["free_living", "laboratory"], optional
        Type of condition, defaults to "free_living".
    check_against : Sequence[Literal["literature", "global"]], optional
        Thresholds to check against, defaults to ("literature", "global").

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame indicating whether each data point is within the thresholds.
    """
    # Check if all required columns are present in the input data
    required_columns = ["cadence_spm", "walking_speed_mps", "stride_length_m", "stride_duration_s"]
    missing_columns = set(required_columns) - set(input_data.columns)
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")

    # Filter thresholds based on cohort
    cohort_thresholds = thresholds.xs(cohort, level="cohort")

    dmo_type_threshold = cohort_thresholds[[*check_against, condition]].agg(["min", "max"], axis=1)

    # Calculate stride length threshold based on height
    max_sl = _max_allowable_stride_length(height_m)
    dmo_type_threshold.loc["stride_length_m", "max"] = max(max_sl, dmo_type_threshold.loc["stride_length_m", "max"])

    # We stack the actual values so that we can perform the comparison in one step
    data_flag = input_data[required_columns].stack()

    # Get the dmo for each row
    dmo_vals = data_flag.index.get_level_values(1)
    # Get the cohort for each row
    per_row_thresholds = dmo_type_threshold.loc[dmo_vals]
    # Check if the value is within the threshold
    data_flag = data_flag.between(
        per_row_thresholds["min"].to_numpy(), per_row_thresholds["max"].to_numpy(), inclusive="both"
    )
    # Unstack the data
    data_flag = data_flag.unstack()  # noqa: PD010

    return data_flag
