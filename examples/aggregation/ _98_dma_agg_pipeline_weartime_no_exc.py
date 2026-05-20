"""
Weekly Aggregation with Pipeline Wear-Time
==========================================

This example shows how to aggregate daily pipeline results to weekly level using
wear-time detected from the pipeline.

The pipeline produces daily aggregated DMOs (one row per recording) including weartime_hours_during_waking.
This example shows how to:
1. Load daily pipeline results from multiple recordings
2. Filter out days with insufficient weartime
3. Aggregate to weekly (per-participant) level

Before you start you need:

- Daily DMO results: CSV file containing aggregated_parameters_ outputs from pipeline runs
  across multiple days/participants. Must include 'weartime_hours_during_waking' column.
- Data should have index/columns: visit_type, participant_id, measurement_date

Use the full path in the `path_config` dictionary below.
"""

from pathlib import Path

path_config = {
    "daily_dmo": ...,
    "outpath": ...,
}

# %%
# Load Daily Aggregated Data
# --------------------------
# Load pre-computed daily aggregations from pipeline (one row per recording/day)
import pandas as pd

daily_aggregated = pd.read_csv(
    path_config["daily_dmo"],
    index_col=["visit_type", "participant_id", "measurement_date"]
)

# %%
# QA: Check data structure
# ------------------------
print(f"Total daily recordings: {len(daily_aggregated)}")
print(f"Unique participants: {daily_aggregated.index.get_level_values('participant_id').nunique()}")
print(f"\nColumns available:\n{daily_aggregated.columns.tolist()}")

# %%
# Weartime Filtering
# ------------------
# Filter out days with insufficient weartime (using pipeline-detected values)
daily_aggregated_filtered = daily_aggregated.query("weartime_hours_during_waking >= 12")


# %%
# QA: Number of recording days per participant
# --------------------------------------------
date_range = (
    daily_aggregated_filtered.index.to_frame()
    .reset_index(drop=True)
    .groupby("participant_id")["measurement_date"]
    .agg(["min", "max", "count"])
    .rename(columns={"count": "n_days", "min": "first", "max": "last"})
    .assign(
        day_diff=lambda df_: (
            df_[["first", "last"]].map(pd.to_datetime).eval("last - first")
        )
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
# We aggregate the daily data to a weekly level and remove DMAs with insufficient data.
weekly_aggregated = (
    daily_aggregated_filtered.drop(columns=["weartime_hours_during_waking"])
    .groupby(["visit_type", "participant_id"])
    .mean(numeric_only=True)
    .assign(
        n_days=daily_aggregated_filtered.groupby(["visit_type", "participant_id"]).size(),
        total_weartime_hours=daily_aggregated_filtered["weartime_hours_during_waking"]
        .groupby(["visit_type", "participant_id"])
        .sum()  # Sum weartime across all valid days
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
    ~weekly_aggregated.columns.isin(round_to_int + ["n_days"])
]
weekly_aggregated[round_to_int] = weekly_aggregated[round_to_int].round()
weekly_aggregated[round_to_three_decimals] = weekly_aggregated[
    round_to_three_decimals
].round(3)

# %%
# Filter: Require at least 3 valid days per participant
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
# Save weartime summary
daily_aggregated[["weartime_hours_during_waking"]].to_csv(outdir / "daily_weartime.csv")
daily_aggregated[daily_aggregated["weartime_hours_during_waking"] < 12].to_csv(
    outdir / "insufficient_weartime_days.csv"
)
