r"""
.. _gsd_evaluation:

GSD Evaluation
==============

This example shows how to apply evaluation algorithms to GSD and thus how to rate the performance of a GSD algorithm.
"""

import pandas as pd
from mobgap.data import LabExampleDataset
from mobgap.gait_sequences import GsdIluz

# %%
# Loading some example data
# -------------------------
# First, we load example data and apply the GSD Iluz algorithm to it.
# However, you can use any other GSD algorithm as well.
# To have a reference to compare the results to, we also load the corresponding ground truth data.
# These steps are explained in more detail in the :ref:`GSD Iluz example <gsd_iluz>`.
from mobgap.utils.conversions import to_body_frame


def load_data():
    lab_example_data = LabExampleDataset(reference_system="INDIP")
    single_test = lab_example_data.get_subset(
        cohort="MS", participant_id="001", test="Test11", trial="Trial1"
    )
    return single_test


def calculate_gsd_iluz_output(single_test_data):
    """Calculate the GSD Iluz output for one sensor from the test data."""
    det_gsd = (
        GsdIluz()
        .detect(
            to_body_frame(single_test_data.data_ss),
            sampling_rate_hz=single_test_data.sampling_rate_hz,
        )
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
# detected and reference gait sequences.
# Note, that there are two different ways to approach this:
#
# 1. We can calculate the for each sample, whether it is correctly detected as gait or not.
# 2. We can check on the level of gait sequences, whether a detected gait sequence matches with a reference gait sequence (by a certain overlap threshold).
#
# In mobgap, we provide functions to calculate both types of performance metrics. Let's start with the first one.
#
# Sample-wise performance evaluation
# ----------------------------------
# To do this, we use the :func:`~gaitlink.gait_sequences.evaluation.categorize_intervals_per_sample` function
# to identify overlapping regions between the detected gait sequences and the reference gait sequences.
# These overlapping regions can then be converted into sample-wise classifications of true positives, false positives, and false negatives.
#
# As function arguments, besides the mandatory detected and reference gait sequences,
# the total number of samples in the recording can be specified as optional parameter.
# If provided, the intervals where no gait sequences are present in the reference and the detected list are also reported.
# Later on, we can then use these categorized intervals to calculate a set of higher-level performance metrics.
#
# As result, a DataFrame containing `start` and `end`  index of the resulting categorized intervals together with
# a `match_type` column that contains the type of match for each interval, i.e. `tp` for true positive, `fp` for false
# positive, and `fn` for false negative.
# These intervals can not be interpreted as gait sequences, but are rather subsequences of the detected gait sequences
# categorizing correctly detected samples (`tp`), falsely detected samples (`fp`), samples from the reference gsd
# list that were not detected (`fn`), and (optionally) samples where no gait sequences are present in both the reference and detected gait sequences (`tn`).
# Note that the tn intervals are not explicitly calculated, but are inferred from the total length of the recording
# (if provided) and from the other intervals, as everything between them is considered as true negative.
from mobgap.gait_sequences.evaluation import categorize_intervals_per_sample

categorized_intervals = categorize_intervals_per_sample(
    gsd_list_detected=detected_gsd_list,
    gsd_list_reference=reference_gsd_list,
    n_overall_samples=len(test_data.data_ss),
)

categorized_intervals

# %%
# Based on the individually categorized tp, fp, fn, and tn intervals, common performance metrics,
# e.g., F1 score, precision, or recall can be calculated.
# For this purpose, the :func:`~gaitlink.gait_sequences.evaluation.calculate_matched_gsd_performance_metrics` function can be used.
# It calculates the metrics based on the "matched" gsd intervals, i.e., the categorized interval list where every entry
# has a match type (tp, fp, fn, tn) assigned.
# Therefore, the function requires to call the :func:`~gaitlink.gait_sequences.evaluation.categorize_intervals_per_sample` function first.
# The categorized intervals can then be passed as an argument
# to :func:`~gaitlink.gait_sequences.evaluation.calculate_matched_gsd_performance_metrics`.
# It returns a dictionary containing the metrics for the specified categorized intervals DataFrame.
# Here, the total number of samples in every match type, precision, recall, F1 score, are always calculated.
# Depending on whether true negatives are present in the categorized intervals,
# specificity, negative predictive value, and accuracy will additionally be reported.
from mobgap.gait_sequences.evaluation import (
    calculate_matched_gsd_performance_metrics,
)

matched_metrics_dict = calculate_matched_gsd_performance_metrics(
    categorized_intervals
)

matched_metrics_dict

# %%
# Furthermore, there is a range of high-level performance metrics that are simply calculated based
# on the overall amount of gait sequences/gait detected in reference and detected data.
# Thus, they can be inferred from the reference and detected gait sequences directly without any intermediate steps
# using the :func:`~gaitlink.gait_sequences.evaluation.calculate_unmatched_gsd_performance_metrics` function.
# As some of the unmatched metrics are reported in seconds, the function requires the sampling frequency of the recorded
# data as an additional argument.
# It returns a dictionary containing all metrics for the specified detected and reference gait sequences.
from mobgap.gait_sequences.evaluation import (
    calculate_unmatched_gsd_performance_metrics,
)

unmatched_metrics_dict = calculate_unmatched_gsd_performance_metrics(
    gsd_list_detected=detected_gsd_list,
    gsd_list_reference=reference_gsd_list,
    sampling_rate_hz=test_data.sampling_rate_hz,
)

unmatched_metrics_dict
# %%
# Direct Gait Sequence Matching
# -----------------------------
# Apart from the performance evaluation methods mentioned above, it might be useful in some cases to identify
# how many and which detected gait sequences reliably match with the ground truth.
#
# This is primarily useful, when further parameters are associated with each gaits sequence, e.g., the gait speed.
# In this case, matching gait sequences that cover the same gait regions allows proper comparison of these parameters.
# For more information on this, see the example on the overall parameter evaluation on Walking-Bout level (TODO).
#
# For this purpose, the :func:`~gaitlink.gait_sequences.evaluation.categorize_intervals` can be used.
# It returns all intervals of the detected gait sequences that overlap with the reference gait sequences by at least a
# given amount.
# The index of the result dataframe indicated the index of the detected gait sequence.
# We can see that with an overlap threshold of 0.7 (70%), three of the six detected gait sequences are considered as
# matches with the reference gait sequences for our example recording.
# Note, that this threshold is enforced in both directions, i.e., the detected gait sequence must overlap with the
# reference gait sequence by at least 70% and vice versa.
# This means that only 1 to 1 matches are possible.
# If multiple detected gait sequences overlap with the same reference gait sequence, only the one with the highest
# overlap is considered as a match.
# If one gait sequence is covered by multiple smaller once, possibly none of them is considered as a match.
from mobgap.gait_sequences.evaluation import categorize_intervals

matches = categorize_intervals(
    gsd_list_detected=detected_gsd_list,
    gsd_list_reference=reference_gsd_list,
    overlap_threshold=0.7,
)

matches

# %%
# Running a full evaluation pipeline
# ----------------------------------
# Instead of manually evaluating and investigating the performance of a GSD algorithm on a single piece of data, we
# often want to run a full evaluation on an entire dataset.
# This can be done using the :class:`~mobgap.gait_sequences.evaluation.GsdEvaluationPipeline` class and some ``tpcp``
# functions.
#
# But let's start with selecting some data.
# We want to use all the simulated real-world walking data from the INDIP reference system (Test11).
simulated_real_world_walking = LabExampleDataset(
    reference_system="INDIP"
).get_subset(test="Test11")

simulated_real_world_walking
# %%
# Now we can use the :class:`~mobgap.gait_sequences.evaluation.GsdEvaluationPipeline` class to directly run a Gsd
# algorithm on a datapoint.
# The pipeline takes care of extracting the required data.
from mobgap.gait_sequences.pipeline import GsdEmulationPipeline

pipeline = GsdEmulationPipeline(GsdIluz())

pipeline.safe_run(simulated_real_world_walking[0]).gs_list_

# %%
# Note, that this did just "run" the pipeline on a single datapoint.
# If we want to run it on all datapoints and evaluate the performance of the algorithm, we can use the
# :func:`~tpcp.validate.validate` function.
#
# For this we need to provide a score function that runs and evaluates the pipeline on a single datapoint.
# We provide a default score function for GSD that calculates all the metrics shown above per datapoint and combined
# accross all gait sequences (i.e. all gait sequences across all datapoints are pooled before metrics are calculated).
from mobgap.gait_sequences.evaluation import gsd_score
from tpcp.validate import validate

evaluation_results = pd.DataFrame(
    validate(pipeline, simulated_real_world_walking, scoring=gsd_score)
)

evaluation_results.drop(
    ["single__raw__reference", "single__raw__detected"], axis=1
).T
# %%
# In addition to the metrics, the method also returns the raw reference and detected gait sequences.
# These can be used for further custom analysis.

evaluation_results["single__raw__reference"][0]

# %%
evaluation_results["single__raw__detected"][0]

# %%
# If you want to calculate additional metrics, you can either create a custom score function or subclass the pipeline
# and overwrite the score function.
#
# Parameter Optimization
# ----------------------
# Simply applying an algorithm to the data for evaluation is often not enough.
# In case, of machine learning algorithms or algorithms with tunable parameters, we might want to optimize these
# parameters to get the best possible performance.
# To avoid overfitting, we can use cross-validation to evaluate the performance of the algorithm on multiple splits of
# the data.
#
# Below we show that procedure by using a simple grid search to optimize the window length of the GSD Iluz algorithm
# and evaluate this approach within a 3-fold cross-validation.
# Per-fold we select the window length leading to the highest precision on the "train set" and evaluate the performance
# on the "test set".
#
# Note, that on a real world dataset, you would likely need to perform a group-wise stratified cross-validation to
# avoid data leakage between multiple trials from the same participant and ensure equal distribution of patient cohorts
# across the folds.
# See the detailed ``tpcp`` examples on these topics.
from sklearn.model_selection import ParameterGrid
from tpcp.optimize import GridSearch
from tpcp.validate import cross_validate

para_grid = ParameterGrid({"algo__window_length_s": [2, 3, 4]})

cross_validate_results = pd.DataFrame(
    cross_validate(
        GridSearch(
            GsdEmulationPipeline(GsdIluz()),
            para_grid,
            return_optimized="precision",
            scoring=gsd_score,
        ),
        simulated_real_world_walking,
        scoring=gsd_score,
        cv=3,
        return_train_score=True,
    )
)

cross_validate_results.drop(
    [
        "test__single__raw__reference",
        "test__single__raw__detected",
        "train__single__raw__reference",
        "train__single__raw__detected",
    ],
    axis=1,
).T

# %%
# In general, it is a good idea to use ``cross_validation`` also for algorithms that do not have tunable parameters.
# This way you can ensure that the performance of the algorithm is stable across different splits of the data, and it
# allows the direct comparison between tunable and non-tunable algorithms.
