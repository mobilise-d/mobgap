from datetime import datetime
from pathlib import Path

import pandas as pd


def _datetime_to_seconds_since_midnight(dt: datetime) -> int:
    return dt.hour * 3600 + dt.minute * 60 + dt.second


def load_weartime_from_daily_mcroberts_report(path: Path, waking_hours: tuple[float, float] = (7, 22)) -> pd.DataFrame:
    """Load the weartime per minute data from the daily MC Roberts report.

    Parameters
    ----------
    path
        The path to the daily MC Roberts report.
    waking_hours
        The waking hours in which the weartime is considered. The first value is the start of the waking hours and the
        second value is the end of the waking hours. Both values are inclusive.
        Note, that we expect that this value is expected in hours as a float from 0 to 24h. Aka 7.5 for 7:30 AM.

    Returns
    -------
    date
        The date of the report.
        This can be used to check if the report is for the correct day.
    total_worn_h
        The total weartime in hours per day.
    total_worn_during_waking_h
        The total weartime in hours per day during the waking hours.

    Notes
    -----
    This only works with the "new" weartime format that is used by McRoberts since 12/2023.
    These weartime reports are generated for all data of the Mobilised clinical validation study.

    """
    if not 0 <= waking_hours[0] < waking_hours[1] <= 24:
        raise ValueError(
            f"Invalid waking hours: {waking_hours}. " f"Expected: (0 <= waking_hours[0] < waking_hours[1] <= 24)"
        )

    waking_hours_as_seconds = (waking_hours[0] * 3600, waking_hours[1] * 3600)

    per_min_report = pd.read_csv(
        path, delimiter=";", parse_dates=["interval_starttime"], usecols=["interval_starttime", "DUR_total_worn"]
    )
    per_min_report["visit_date"] = per_min_report["interval_starttime"].dt.date
    per_min_report = per_min_report.set_index("interval_starttime")
    result = (
        per_min_report.groupby("visit_date")
        .agg(
            total_worn_h=("DUR_total_worn", "sum"),
            total_worn_during_waking_h=(
                "DUR_total_worn",
                lambda x: x[
                    (x.index.time >= datetime.fromtimestamp(waking_hours_as_seconds[0]).time())
                    & (x.index.time < datetime.fromtimestamp(waking_hours_as_seconds[1]).time())
                ].sum(),
            ),
        )
        .fillna(0)
    )

    return result / 3600
