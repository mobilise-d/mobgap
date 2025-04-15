"""
.. _threshold_check_example:

Threshold Checker
=================

This example shows how to use the functions ``apply_thresholds`` to check if a DMO falls within the thresholds for a
specific cohort and height.
"""

import pandas as pd
from mobgap import PROJECT_ROOT
from mobgap.aggregation import apply_thresholds, get_mobilised_dmo_thresholds

# %%
# Selecting Example Data
# ----------------------
# We use some example DMO data.
# Note, that the ``apply_thresholds`` function can only be used with the data of a single participant at the time,
# as meta-data such as the participant's height is required.
# Luckily, the all the example data is from the same participant, just recorded at different times.

DATA_PATH = PROJECT_ROOT / "example_data/original_results/mobilised_aggregator"

data = pd.read_csv(
    DATA_PATH / "aggregation_test_input.csv", index_col=0
).set_index(["visit_type", "participant_id", "measurement_date", "wb_id"])
data.head()

# %%
# Load Thresholds
# ---------------
thresholds = get_mobilised_dmo_thresholds()
thresholds

# %%
# Apply Thresholds
# ----------------
data_mask = apply_thresholds(
    data,
    thresholds,
    cohort="CHF",
    height_m=1.75,
    measurement_condition="free_living",
)
data_mask.head()

# %%
# We can see that the output has exactly the same structure as the input data, but with boolean values indicating if
# the DMO falls within the thresholds or not.
# Columns for which no thresholds were provided are simply NaN, indicating that no filtering was applied.
# It is up to you how you want to use this information.

# %%
# This output data can be used to filter the input data for DMOs that fall within the thresholds.
# This can be done in combination with the aggregation algorithm to only include DMOs that fall within the thresholds.
# See the :ref:`mobilised_aggregator example <mobilised_aggregator_example>` for more information.
