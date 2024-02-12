"""
Threshold Checker
====================

This example shows how to use the functions threshold_check to check if the points fall within thresholds or not.
"""

import pandas as pd

from gaitlink.aggregation._threshold_check import apply_thresholds, get_mobilised_dmo_thresholds

# Absolute path to the input file
input_data = pd.read_csv("gaitlink\\example_data\\original_results\\mobilised_aggregator\\aggregation_test_input.csv")

# Extract Thresholds
thresholds_data = get_mobilised_dmo_thresholds()
output_data = apply_thresholds(input_data, thresholds_data, cohort="CHF", height_m=1.75)
print(output_data)

# Else save the file
# output_data.to_csv(filename)
