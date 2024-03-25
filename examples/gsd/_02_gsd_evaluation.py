r"""
.. _gsd_evaluation:

GSD Evaluation
==============

This example shows how to apply evaluation algorithms to GSD and thus how to rate the performance of a GSD algorithm.
"""

from mobgap.data import LabExampleDataset
from mobgap.gsd import GsdIluz

# %%
# Loading some example data
# -------------------------
# First, we load example data and apply the GSD Iluz algorithm to it.
# However, you can use any other GSD algorithm as well.
# To have a reference to compare the results to, we also load the corresponding ground truth data.
# These steps are explained in more detail in the :ref:`GSD Iluz example <gsd_iluz>`.


def load_data():
    lab_example_data = LabExampleDataset(reference_system="INDIP")
    single_test = lab_example_data.get_subset(cohort="MS", participant_id="001", test="Test11", trial="Trial1")
    return single_test


def calculate_gsd_iluz_output(single_test_data):
    """Calculate the GSD Iluz output for one sensor from the test data."""
    det_gsd = (
        GsdIluz()
        .detect(single_test_data.data["LowerBack"], sampling_rate_hz=single_test_data.sampling_rate_hz)
        .gs_list_
    )
    return det_gsd


def load_reference(single_test_data):
    """Load the reference gait sequences from the test data."""
    ref_gsd = single_test_data.reference_parameters_.wb_list
    return ref_gsd


test_data = load_data()
detected_gsd_list = calculate_gsd_iluz_output(test_data)
reference_gsd_list = load_reference(test_data)

# %% The resulting data is a simple dataframe containing one gait sequence per row,
# that is characterized by its start and end index in samples.
detected_gsd_list

# %%
reference_gsd_list

# %%
# Validation of algorithm output against a reference
# --------------------------------------------------
# Let's quantify how the algorithm output compares to the reference labels.
# To gain a detailed insight into the performance of the algorithm, we can look into the individual matches between the
# detected and reference initial contacts.
# To do this, we use the :func:`~gaitlink.gsd.evaluation.categorize_intervals` function
# to identify overlapping regions between the detected gait sequences and the reference gait sequences.
# As function arguments, besides the mandatory detected and reference gait sequences,
# the total number of samples in the recording can be specified as optional parameter.
# If provided, the intervals where no gait sequences are present in the reference and the detected list are
# also reported.
# Later on, we can then use these categorized intervals to calculate a range of higher-level performance metrics.
#
# As result, a DataFrame containing `start` and `end`  index of the resulting categorized intervals together with
# a `match_type` column that contains the type of match for each interval, i.e. `tp` for true positive, `fp` for false
# positive, and `fn` for false negative.
# These intervals can not be interpreted as gait sequences, but are rather subsequences of the detected gait sequences
# categorizing correctly detected samples (`tp`), falsely detected samples (`fp`), samples from the reference gsd
# list that were not detected (`fn`), and (optionally) samples where no gait sequences are present in both the reference
# and detected gait sequences (`tn`).
# Note that the tn intervals are not explicitly calculated, but are inferred from the total length of the recording
# (if provided) and from the other intervals, as everything between them is considered as true negative.
from mobgap.gsd.evaluation import categorize_intervals

categorized_intervals = categorize_intervals(
    gsd_list_detected=detected_gsd_list,
    gsd_list_reference=reference_gsd_list,
    n_overall_samples=len(test_data.data["LowerBack"]),
)

categorized_intervals

# %%
# Based on the individually categorized tp, fp, fn, and tn intervals, common performance metrics,
# e.g., F1 score, precision, or recall can be calculated.
# For this purpose, the :func:`~gaitlink.gsd.evaluation.calculate_matched_gsd_performance_metrics` function can be used.
# It calculates the metrics based on the 'matched' gsd intervals, i.e., the categorized interval list where every entry
# has a match type (tp, fp, fn, tn) assigned.
# Therefore, the function requires to call the :func:`~gaitlink.gsd.evaluation.categorize_intervals` function first.
# The categorized intervals can then be passed as an argument
# to :func:`~gaitlink.gsd.evaluation.calculate_matched_gsd_performance_metrics`.
# It returns a dictionary containing the metrics for the specified categorized intervals DataFrame.
# Here, the total number of samples in every match type, precision, recall, F1 score, are always calculated.
# Depending on whether true negatives are present in the categorized intervals,
# specificity, negative predictive value, and accuracy will additionally be reported.
from mobgap.gsd.evaluation import calculate_matched_gsd_performance_metrics

matched_metrics_dict = calculate_matched_gsd_performance_metrics(categorized_intervals)

matched_metrics_dict

# %%
# Furthermore, there is a range of high-level performance metrics that is calculated based
# on the overall amount of gait sequences in reference and detected data.
# Thus, they can be inferred from the reference and detected gait sequences directly without any intermediate steps
# using the :func:`~gaitlink.gsd.evaluation.calculate_unmatched_gsd_performance_metrics`
# function.
# As some of the unmatched metrics are reported in seconds, the function requires the sampling frequency of the recorded
# data as an additional argument.
# It requires specifying the sampling frequency of the recorded data (to calculate the duration errors in seconds)
# and returns a dictionary containing all metrics for the specified detected and reference gait sequences.
from mobgap.gsd.evaluation import calculate_unmatched_gsd_performance_metrics

unmatched_metrics_dict = calculate_unmatched_gsd_performance_metrics(
    gsd_list_detected=detected_gsd_list,
    gsd_list_reference=reference_gsd_list,
    sampling_rate_hz=test_data.sampling_rate_hz,
)

unmatched_metrics_dict
# %%
# Apart from the performance evaluation methods mentioned above, it might be useful in some cases to identify
# how many and which detected gait sequences reliably match with the ground truth.
# This can be helpful for instance when developing a new detection algorithm or when investigating aggregated parameters
# from within the respective gait sequences.
# For this purpose, the :func:`~gaitlink.gsd.evaluation.find_matches_with_min_overlap` can be used.
# It returns all intervals of the detected gait sequences that overlap with the reference gait sequences by at least a
# given amount.
# The index of the result dataframe indicated the index of the detected gait sequence.
# We can see that with an overlap threshold of 0.7 (70%), three of the six detected gait sequences are considered as
# matches with the reference gait sequences for our example recording.
# The remaining ones either contain too many false positive and/or false negative samples.
from mobgap.gsd.evaluation import find_matches_with_min_overlap

matches = find_matches_with_min_overlap(
    gsd_list_detected=detected_gsd_list,
    gsd_list_reference=reference_gsd_list,
    overlap_threshold=0.7,
)

matches
