"""

.. sdmo_aggregator_example:

Signal-based digital mobility outcomes (SDMO) Aggregator
========================================================

This example shows how to use the :class:`.SDMOAggregator` class to aggregate SDMOs over multiple walking bouts.
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
from mobgap.aggregation import SDMOAggregator

DATA_PATH = PROJECT_ROOT / "example_data/original_results/sdmo_aggregator"

data = (
    pd.read_csv(DATA_PATH / "sdmo_aggregation_test_input.csv", index_col=0)
    .set_index(["visit_type", "participant_id", "measurement_date", "wb_id"])
    .rename(columns={"n_steps": "n_raw_initial_contacts"})
)
data.head()

# %%
# Performing the aggregation
# --------------------------
# The :class:`.SDMOAggregator` is used to aggregate the input data over several walking bouts, e.g., over all
# walking bouts from one participant, or over all walking bouts per participant and day, week, or other criteria.
# The data is grouped using additional columns in the input data, which are not used for the aggregation itself.
# In this example, the data is grouped by participant (`subject_code`) and day (`visit_date`).
# Although default aggregations for the  :class:`.MobilisedPipeline` are provided, this class can be used to perform
# any aggregations by providing the `duration_filters` (for walking bout duration filtering) and `metrics` (for
# calculating statistics)
agg = SDMOAggregator(
    **dict(
        SDMOAggregator.PredefinedParameters.cvs_sdmo_data,
    )
)
agg.aggregate(data)
