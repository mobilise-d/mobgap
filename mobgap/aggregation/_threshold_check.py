from collections.abc import Sequence
from importlib.resources import files
from typing import Optional

import numpy as np
import pandas as pd
from typing_extensions import Literal

from mobgap.consts import GRAV_MS2


def get_mobilised_dmo_thresholds() -> pd.DataFrame:
    """Load the mobilised DMO thresholds.

    They can be used together with the :func:`apply_thresholds` function to check if a DMO falls within the thresholds.

    Returns
    -------
    pd.DataFrame
        Mobilised DMO thresholds with a multi-level index.
        This is a dataframe with a multi-level index, where the first level is DMO type and the second level is the
        cohort.
        The columns are also a multi-level index with the first level being the "type" of threshold, indicating from
        where the threshold was derived (e.g. "literature", "global", "free_living", "laboratory") and the second
        level being the actual threshold type ("min", "max").

    """
    with (
        files("mobgap") / "aggregation/_dmo_thresholds/official_mobilised_dmo_thresholds.csv"
    ).open() as threshold_data:
        thresholds = pd.read_csv(threshold_data, header=0, index_col=[0, 1])
    thresholds = thresholds.bfill().drop("ALL", level=1)
    thresholds.columns = pd.MultiIndex.from_tuples(
        [c.rsplit("_", 1) for c in thresholds.columns], names=["condition", "threshold_type"]
    )
    return thresholds


def _max_allowable_stride_length(height_m: float) -> float:
    """

    Calculate the maximum allowable stride length based on the participant's height.

    Parameters
    ----------
    height_m
        Participant's height in meters.

    Returns
    -------
    float
        Maximum allowable stride length.
    """
    leg_length = 0.53 * height_m  # Winter 2009
    froude_number = 1
    v_max = np.sqrt(froude_number * GRAV_MS2 * leg_length)  # Ivanenko 2007
    max_vertical_displacement = 0.038 * v_max**2  # Miff et al. (2000)
    max_sl = 2 * 2 * np.sqrt(2 * leg_length * max_vertical_displacement - max_vertical_displacement**2)  # Zijlstra
    return max_sl


def apply_thresholds(
    input_data: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    cohort: str,
    height_m: Optional[float],
    measurement_condition: Literal["free_living", "laboratory"],
    check_against: Sequence[Literal["literature", "global"]] = ("literature", "global"),
) -> pd.DataFrame:
    """
    Apply DMO thresholds to input data and return a DataFrame check whether each data point is within the threshold.

    All thresholds are inclusive to the maximum and minimum values.
    To understand how the ``thresholds`` DataFrame is structured, please refer to the documentation of the function
    :func:`get_mobilised_dmo_thresholds`.

    In addition to the provided thresholds, the function also calculates the maximum allowable stride length based on
    the participant's height using a simple model (See Notes).
    If you don't want to use this feature, you can set the height to None.

    Parameters
    ----------
    input_data
        Input data containing columns 'cadence_spm', 'walking_speed_mps', 'stride_length_m', 'stride_duration_s'.
    thresholds
        DMO thresholds data.
        This is a dataframe with a multi-level index, where the first level is DMO type and the second level is the
        cohort.
        The columns are also a multi-level index with the first level being the "type" of threshold, indicating from
        where the threshold was derived (e.g. "literature", "global", "free_living", "laboratory") and the second level
        being the actual threshold type ("min", "max").
    cohort
        Cohort identifier for filtering thresholds.
    height_m
        Participant's height in meters.
        Set to None if you don't want to use the height based stride length threshold.
    measurement_condition
        The measurement condition the data was recorded in.
        This is used to select one of the threshold columns.
    check_against
        Additional thresholds to check against in addition to the measured condition thresholds.
        Defaults to ("literature", "global").
        The exact options available might depend on the thresholds provided in the input data.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame indicating whether each data point is within the thresholds.
        For values in columns, for which no thresholds were provided this df contains NaN values indicating no checks
        were applied.

    Notes
    -----
    The height threshold uses an equation from [1]_ to calculate the leg length from the participant's height and then
    estimates the maximal allowable gait speed according to [2]_ considering a Froude number of 1.
    Then we use [3]_ to calculate the maximal vertical displacement of the center of mass during walking at that speed.
    The maximal allowable stride length is then calculated using the inverted pendulum model introduced by
    Zijlstra et al..

    .. [1] Biomechanics and Motor Control of Human Movement, Winter 2009
    .. [2] Ivanenko YP, Cappellini G, Dominici N, Poppele RE, Lacquaniti F. Modular control of limb movements during
       human locomotion. J Neurosci. 2007 Oct 10;27(41):11149-61. doi: 10.1523/JNEUROSCI.2644-07.2007. PMID: 17928457;
       PMCID: PMC6672838.
    .. [3] S. C. Miff, S. A. Gard and D. S. Childress, "The effect of step length, cadence, and walking speed on the
       trunk's vertical excursion," Proceedings of the 22nd Annual International Conference of the IEEE Engineering in
       Medicine and Biology Society (Cat. No.00CH37143), Chicago, IL, USA, 2000, pp. 155-158 vol.1,
       doi: 10.1109/IEMBS.2000.900692.


    """
    # Check if all required columns are present in the input data
    required_columns = ["cadence_spm", "walking_speed_mps", "stride_length_m", "stride_duration_s"]
    missing_columns = set(required_columns) - set(input_data.columns)
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")

    # Filter thresholds based on cohort
    cohort_thresholds = thresholds.xs(cohort, level="cohort")

    dmo_type_threshold = cohort_thresholds[[*check_against, measurement_condition]].agg(["min", "max"], axis=1)

    # Calculate stride length threshold based on height
    if height_m is not None:
        max_sl = _max_allowable_stride_length(height_m)
        dmo_type_threshold.loc["stride_length_m", "max"] = max(max_sl, dmo_type_threshold.loc["stride_length_m", "max"])

    # We stack the actual values so that we can perform the comparison in one step
    data_flag = input_data[required_columns].stack()

    # Get the dmo for each row
    dmo_vals = data_flag.index.get_level_values(-1)
    # Get the cohort for each row
    per_row_thresholds = dmo_type_threshold.loc[dmo_vals]
    # Check if the value is within the threshold
    data_flag = data_flag.between(
        per_row_thresholds["min"].to_numpy(), per_row_thresholds["max"].to_numpy(), inclusive="both"
    )
    # Unstack the data
    data_flag = data_flag.unstack()  # noqa: PD010

    data_flag = data_flag.reindex(input_data.index).reindex(input_data.columns, axis=1)

    return data_flag
