"""
CVS Official Aggregation Script
===============================

This example shows how the raw CVS per walking bout results are combined with the weartime reports and aggregated to a
daily and weekly level.

For more details on the individual steps see the other aggregation examples.
This example should primarily serve as an easy to use script for people that want to run/replicate the CVS aggregation.

Before you start you need 3 things:

- DMO data:A csv file obtained from the Mobilise-D datawarehouse that contains the data of one CVS measurement time points
  (e.g. T1). Example file name: `cvs-T3-wb-dmo-14-05-2024.csv`
- PID-Map: A mapping between patient ids and measurement sites (short pid-map). Example file name:
  `study-instances-Cohort Site-2023-08-08h22m09s48.csv`
- Weartime: The minute-by-minute weartime reports obtained by McRoberts. This is a folder with a large amount of csv files.
  In addition to the csv files, we also expect a file with the pattern `CVS-wear-compliance-*.xlsx` in the same folder.
  This is used to map the measurement-ids to the correct weartime reports.

Use the full path to these files/folders in the `path_config` dictionary below.
The "outpath" key in the `path_config` dictionary specifies the folder where the aggregated data should be saved.
"""

from pathlib import Path

path_config = {
    "dmo": ...,
    "pid_map": ...,
    "weartime_reports": ...,
    "outpath": ...,
}

# %%
# Load the data
# -------------
# We create a new dataset instance that contains all the data and allows use to load and query it efficiently.
from joblib import Memory
from mobgap.data import MobilisedCvsDmoDataset

cache = Memory(".cache")
ds = MobilisedCvsDmoDataset(
    path_config["dmo"],
    path_config["pid_map"],
    memory=cache,
    weartime_reports_base_path=path_config["weartime_reports"],
)

# %%
# QA: Check for duplicated DMOs
# -----------------------------
# We check if there are any duplicated DMOs in the data.
# If yes, we create a warning and save a file for further investigation.
#
# .. note:: The initial loading of the data might take some time
data = ds.data
duplicated_dmos = data.index.to_frame()[data.index.to_frame().duplicated()]
if not duplicated_dmos.empty:
    print("Warning: Duplicated DMOs found. Saving to 'duplicated_dmos.csv'")
    duplicated_dmos.to_csv(Path(path_config["outpath"]) / "duplicated_dmos.csv")

# %%
# Daily agg
# ---------
# We aggregate the data per day for each participant.
# We then merge the results with the weartime reports to later filter out days that have not enough weartime.
from mobgap.aggregation import MobilisedAggregator

daily_agg = MobilisedAggregator(
    **MobilisedAggregator.PredefinedParameters.cvs_dmo_data
)
daily_agg.aggregate(data, data_mask=ds.data_mask)

# %%
# To exactly match the output format of the original Mobilise-D R-Script, we round the output to 3 decimal places and
# convert the stride length values to cm.
agg_values = daily_agg.aggregated_data_
agg_values[["strlen_1030_avg", "strlen_30_avg"]] *= 100
agg_values = agg_values.round(3)

# %%
# Further, we express the variance parameters in "%".
agg_values[
    [
        "wbdur_all_var",
        "cadence_all_var",
        "strdur_all_var",
        "ws_30_var",
        "strlen_30_var",
    ]
] *= 100


# %%
# Merge with weartime.
# .. note:: The initial loading of the weartime data will take some time.
#           After it is loaded once, a new file `daily_weartime_pre_computed.csv` will be created in the weartime
#           folder and used as cache for subsequent loads.
daily_weartime = ds.weartime_daily
daily_aggregated = agg_values.merge(
    daily_weartime, left_index=True, right_index=True
)

# %%
# We drop the  "wb_1030_sum" column, because it was not officially verified as a DMO
daily_aggregated = daily_aggregated.drop(columns=["wb_1030_sum"])

# %%
# Weartime Filtering
# ------------------
# We filter out days that do not have enough weartime (or not weartime info at all).
daily_aggregated_filtered = daily_aggregated.query(
    "total_worn_during_waking_h > 12"
)


# %%
# QA: Number of recording days per participant
# --------------------------------------------
# We check the number of recording days per participant.
import pandas as pd

date_range = (
    daily_aggregated_filtered.index.to_frame()
    .reset_index(drop=True)
    .groupby("participant_id")["measurement_date"]
    .agg(["min", "max", "count"])
    .rename(columns={"count": "n_days", "min": "first", "max": "last"})
    .assign(
        day_diff=lambda df_: df_[["first", "last"]]
        .map(pd.to_datetime)
        .eval("last - first")
    )
)

date_range.to_csv(Path(path_config["outpath"]) / "measurement_ranges.csv")

if not (over_7 := date_range.query("n_days > 7")).empty:
    print(
        f"Warning: {len(over_7)} Participants with more than 7 valid recording days found."
    )

if not (over_6_diff := date_range.query("day_diff > '6 days'")).empty:
    print(
        f"Warning: {len(over_6_diff)} Participants with more than 6 days between first and last valid recording day "
        "found."
    )


# %%
# Weekly Aggregation
# ------------------
# We aggregate the daily data to a weekly level and remove weeks with insufficient data.
weekly_aggregated = (
    daily_aggregated_filtered.drop(
        columns=["total_worn_h", "total_worn_during_waking_h"]
    )
    .groupby(["visit_type", "participant_id"])
    .mean(numeric_only=True)
    .assign(
        n_days=daily_aggregated_filtered["walkdur_all_sum"]
        .groupby(["visit_type", "participant_id"])
        .count()
    )
)

# %%
# Formatting:
round_to_int = [
    "walkdur_all_sum",
    "turns_all_sum",
    "wb_all_sum",
    "wb_10_sum",
    "wb_30_sum",
    "wb_60_sum",
]
round_to_three_decimals = weekly_aggregated.columns[
    ~weekly_aggregated.columns.isin(round_to_int)
]
weekly_aggregated[round_to_int] = weekly_aggregated[round_to_int].round()
weekly_aggregated[round_to_three_decimals] = weekly_aggregated[
    round_to_three_decimals
].round(3)

# %%
# Filtering:
weekly_aggregated_filtered = weekly_aggregated[weekly_aggregated["n_days"] >= 3]


# %%
# Export
# ------
from datetime import datetime

current_date = datetime.now().strftime("%Y_%m_%d")

outdir = Path(path_config["outpath"]) / f"export_{current_date}"
outdir.mkdir(exist_ok=True, parents=True)
# %%
daily_aggregated.add_suffix("_d").to_csv(outdir / "daily_agg_all.csv")
daily_aggregated_filtered.add_suffix("_d").to_csv(
    outdir / "daily_agg_filtered.csv"
)
weekly_aggregated.add_suffix("_w").to_csv(outdir / "weekly_agg_all.csv")
weekly_aggregated_filtered.add_suffix("_w").to_csv(
    outdir / "weekly_agg_filtered.csv"
)
daily_weartime.to_csv(outdir / "daily_weartime.csv")
daily_weartime[
    daily_weartime["total_worn_h"].isna()
].index.to_frame().reset_index(drop=True).to_csv(
    outdir / "missing_weartime.csv"
)
