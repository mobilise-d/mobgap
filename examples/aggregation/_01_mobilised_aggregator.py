"""

.. _mobilised_aggregator_example:

Mobilised Aggregator
====================

This example shows how to use the :class:`.MobilisedAggregator` class to aggregate DMOs over multiple walking bouts.
"""

# %%
# Loading some example data
# -------------------------
# .. note :: This data is randomly generated and not physiologically meaningful. However, it has the same structure as
#    any other typical input dataset for the :class:`.MobilisedAggregator`.
#
# The input data for the aggregator is a :class:`pandas.DataFrame` with one row for every walking bout.
# The columns contain the DMO parameters estimated for each walking bout, such as duration, stride length, etc.

import pandas as pd
from mobgap import PROJECT_ROOT
from mobgap.aggregation import MobilisedAggregator

DATA_PATH = PROJECT_ROOT / "example_data/original_results/mobilised_aggregator"

data = (
    pd.read_csv(DATA_PATH / "aggregation_test_input.csv", index_col=0)
    .set_index(["visit_type", "participant_id", "measurement_date", "wb_id"])
    .rename(columns={"n_steps": "n_raw_initial_contacts"})
)
data.head()

# %%
# Furthermore, the aggregator allows to provide a data mask, which is a boolean :class:`pandas.DataFrame` with the same
# dimensions as the input data.
# The data mask indicates which DMOs of the input data should be used for the aggregation (marked as True) and which
# should be ignored (marked as False).
#
# For this example, we create this mask by applying the "standard" thresholds from Mobilise-D to the data.
# To learn more about this see the example :ref:`threshold_check example <threshold_check_example>`.
#
# .. note :: It is only possible to use the ``apply_thresholds`` function here, as all the example data is from the same
#    participant.
#    As some thresholds are cohort or height specific, you would have to apply the thresholds for each participant data
#    separately.
from mobgap.aggregation import apply_thresholds, get_mobilised_dmo_thresholds

thresholds = get_mobilised_dmo_thresholds()
# Note: The height is "artificially" set to 1.75m, as the example data does not contain this information.
data_mask = apply_thresholds(
    data,
    thresholds,
    cohort="HA",
    height_m=1.75,
    measurement_condition="free_living",
)

# %%
# Performing the aggregation
# --------------------------
# The :class:`.MobilisedAggregator` is now used to aggregate the input data over several walking bouts, e.g., over all
# walking bouts from one participant, or over all walking bouts per participant and day, week, or other criteria.
# The data is grouped using additional columns in the input data, which are not used for the aggregation itself.
# In this example, the data is grouped by participant (`subject_code`) and day (`visit_date`).
agg = MobilisedAggregator(
    **dict(
        MobilisedAggregator.PredefinedParameters.cvs_dmo_data,
        use_original_names=False,
    )
)
agg.aggregate(data, wb_dmos_mask=data_mask)

# %%
# The resulting :class:`pandas.DataFrame` containing the aggregated data contains one row for every group.
# In this case, there is only one participant and day, so the resulting dataframe contains only one row.
agg_data = agg.aggregated_data_
agg_data

# %%
# .. warning:: To exactly match the expected output of the original Mobilise-D R-Script, the two stride length
#              parameters would need to be converted to cm and all values rounded to 3 decimals.
#              This is not done in the Python implementation to be consistent with the units across the entire package.
#
# Comparison with R aggregation script
# ------------------------------------
# The outputs of this aggregation algorithm are analogous to the outputs of the original Mobilise-D R-Script, using
# the same duration filters and aggregation metrics.
# However, there can be small differences in the second/third decimal place range in the results. This is due to
# different outputs of the quantile function in Python and R.
# Furthermore, the parameter "strlen_30_var" is converted to cm for consistency, while it is in m in the original
# R-Script.
# By grouping the data by participant and day, the results the Daily Aggregations of the original R-Script are
# retrieved.
# To get the Weekly Aggregations, the Daily results are averaged over all recording days per participant and rounded
# depending on the aggregation metric. Obviously, in this example, the results are identical to the Daily Aggregations,
# as there is only data from one day contained.
weekly_agg = (
    agg.aggregated_data_.groupby("participant_id")
    .mean(numeric_only=True)
    .reset_index()
)
round_to_int_original_cols = [
    "wbsteps_all_sum",
    "turns_all_sum",
    "wb_all_sum",
    "wb_10_sum",
    "wb_30_sum",
    "wb_60_sum",
]
round_to_int_new_cols = [
    "wb_all__n_raw_initial_contacts__sum",
    "wb_all__n_turns__sum",
    "wb_all__count",
    "wb_10__count",
    "wb_30__count",
    "wb_60__count",
]

round_to_int = (
    round_to_int_original_cols
    if agg.use_original_names
    else round_to_int_new_cols
)

round_to_three_decimals = weekly_agg.columns[
    ~weekly_agg.columns.isin(round_to_int)
]
weekly_agg[round_to_int] = weekly_agg[round_to_int].round()
weekly_agg[round_to_three_decimals] = weekly_agg[round_to_three_decimals].round(
    3
)
weekly_agg
