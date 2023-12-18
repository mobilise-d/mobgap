"""
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

from gaitlink import PACKAGE_ROOT
from gaitlink.aggregation import MobilisedAggregator

DATA_PATH = PACKAGE_ROOT.parent / "example_data/original_results/mobilised_aggregator"

data = pd.read_csv(DATA_PATH / "aggregation_test_input.csv", index_col=0)
data.head()

# %%
# Furthermore, the aggregator allows to provide a data mask, which is a boolean :class:`pandas.DataFrame` with the same
# dimensions as the input data.
# The data mask indicates which DMOs of the input data should be used for the aggregation (marked as True) and which
# should be ignored (marked as False).
# TODO: Fix the format of the data mask to actually match the index of the input data
data_mask = pd.read_csv(DATA_PATH / "aggregation_test_data_mask.csv", index_col=0)
data_mask.head()

# %%
# Performing the aggregation
# --------------------------
# The :class:`.MobilisedAggregator` is now used to aggregate the input data over several walking bouts, e.g., over all
# walking bouts from one participant, or over all walking bouts per participant and day, week, or other criteria.
# The data is grouped using additional columns in the input data, which are not used for the aggregation itself.
# In this example, the data is grouped by participant (`subject_code`) and day (`visit_date`).
agg = MobilisedAggregator(groupby=["subject_code", "visit_date"])
agg.aggregate(data, data_mask=data_mask)

# %%
# The resulting :class:`pandas.DataFrame` containing the aggregated data contains one row for every group.
# In this case, there is only one participant and day, so the resulting dataframe contains only one row.
agg.aggregated_data_

# %%
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
weekly_agg = agg.aggregated_data_.groupby("subject_code").mean(numeric_only=True).reset_index()
round_to_int = ["steps_all_sum", "turns_all_sum", "wb_all_sum", "wb_10_sum", "wb_30_sum", "wb_60_sum"]
round_to_three_decimals = weekly_agg.columns[~weekly_agg.columns.isin(round_to_int)]
weekly_agg[round_to_int] = weekly_agg[round_to_int].round()
weekly_agg[round_to_three_decimals] = weekly_agg[round_to_three_decimals].round(3)
weekly_agg
