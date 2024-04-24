r"""
.. _gsd_evaluation_parameter:

GSD Evaluation based on DMO outcomes
====================================

This example shows how to evaluate the performance of the gait sequence detection on gait sequence level.
This is done by comparing the DMOs estimated based on the detected gait sequences with the DMOs estimated based on the
ground truth gait sequences.
"""

import numpy as np

# %%
# Loading some example data
# -------------------------
# To apply the DMO-level evaluation, we require a set of detected gait sequences as well as the corresponding reference
# gait sequences.
# They can be obtained from any GSD algorith of your choice, see the :ref:`GSD Iluz example <gsd_iluz>` example
# for a possible implementation.
# Furthermore, we need the DMOs estimated based on the detected and reference gait sequences.
#
# TODO: refer to suitable example
#
# For this example, we will load some mock data for simplicity.
#
# .. note :: This data is randomly generated and not physiologically meaningful. However, it has the same structure as
#    any other typical input data for this evaluation.
import pandas as pd

from mobgap import PACKAGE_ROOT

DATA_PATH = PACKAGE_ROOT.parent / "example_data/dmo_data/dummy_dmo_data"

detected_gs = pd.read_csv(DATA_PATH / "detected_gsd.csv").set_index(
    ["visit_type", "participant_id", "measurement_date", "wb_id"]
)
detected_dmo = pd.read_csv(DATA_PATH / "detected_dmo_data.csv").set_index(
    ["visit_type", "participant_id", "measurement_date", "wb_id"]
)

reference_gs = pd.read_csv(DATA_PATH / "reference_gsd.csv").set_index(
    ["visit_type", "participant_id", "measurement_date", "wb_id"]
)
reference_dmo = pd.read_csv(DATA_PATH / "reference_dmo_data.csv").set_index(
    ["visit_type", "participant_id", "measurement_date", "wb_id"]
)

# %%
# The gsd data is a simple dataframe containing one gait sequence per row,
# that is characterized by its start and end index in samples.
# The index contains multiple levels, including the visit type, participant_id, measurement day, and gait sequence id.
detected_gs

# %%
reference_gs

# %% The corresponding DMO data is a dataframe with one row per gait sequence.
# The index is identical to the index of the gait sequence data, i.e., refers to the same gait sequences.
detected_dmo

# %%
reference_dmo

# %%
# Extract Matching Detected Gait Sequences
# ----------------------------------------
# Next, we need to extract the gait sequences from the detected data that match the reference gait sequences to be able
# to compare the DMOs gait sequence by gait sequence.
#
# Our example data only contains data from a single visit type, participant, and recording day.
# However, normally the data structure would be more complex, containing several participants, trials,
# and recording days.
# In this case, we want to only match gait sequences within the same trial, day, and participant.
# For this, we need to group the detected and reference initial contacts by those index levels.
# This can be done using the :func:`~mobgap.utils.array_handling.create_multi_groupby` helper function.

from mobgap.utils.array_handling import create_multi_groupby

per_trial_participant_day_grouper = create_multi_groupby(
    detected_gs, reference_gs, groupby=["visit_type", "participant_id", "measurement_date"]
)

print(per_trial_participant_day_grouper)

# %%
# This provides us with a groupby-object that is similar to the normal pandas groupby-object that can be created from a
# single dataframe.
# The ``MultiGroupBy`` object allows us to apply a function to each group across all dataframes.
# Here, we will extract the detected gait sequences that match reference gait sequences within the same group.
#
# For this purpose, the :func:`~mobgap.gsd.evaluation.categorize_matches_with_min_overlap` function is used.
# It classifies every gait sequence in the data either as true positive (TP), false positive (FP),
# or false negative (TP).
# For the TP gait sequences, the corresponding reference gait sequence is assigned.
# For this application, only the matching gait sequences, i.e., the TPs, are of interest.
# If you are interested in the FPs or FNs, have a look at the general :ref:`GSD evaluation example <gsd_evaluation>`.
# The `overlap_threshold` parameter defines the minimum overlap between the detected and reference gait sequences to be
# considered a match.
# It can be chosen according to your needs, whereby a value closer to 0.5 will yield more matches
# than a value closer to 1.

from mobgap.gsd.evaluation import categorize_matches_with_min_overlap

gs_tp_fp_fn = create_multi_groupby(
    detected_gs, reference_gs, groupby=["visit_type", "participant_id", "measurement_date"]
).apply(
    lambda det, ref: categorize_matches_with_min_overlap(
        gsd_list_detected=det, gsd_list_reference=ref, overlap_threshold=0.8, multiindex_warning=False
    )
)
print(gs_tp_fp_fn)


# %%
# Based on the positive matches,
# we can now extract the DMO data from detected and reference data that is to be compared.
# To combine this data, we use the :func:`~mobgap.gsd.evaluation.get_matching_gs` function.

from mobgap.gsd.evaluation import get_matching_gs

gs_matches = get_matching_gs(metrics_detected=detected_dmo, metrics_reference=reference_dmo, matches=gs_tp_fp_fn)
print(gs_matches)

# %%
# Estimate Errors in DMO data
# ----------------------------
# The DMO data can now be compared gait sequence by gait sequence.
# For this purpose, the :func:`~mobgap.gsd.evaluation.assign_error_metrics` is used.
# As input, it receives the matching DMO data and a list of DMOs that should be evaluated.
# Note that those DMOs must be named exactly as in the dmo data columns.
# For some DMOs, it might occur that the reference value is 0, which would lead to a division by zero when calculating
# the relative error. In this case this happens for the `n_turns` parameter.
# Per default, the function raises a warning when zero division occurs and sets the relative error to NaN.
# Here, we suppress the warning by setting the `zero_division_hint` parameter to `np.nan`.

from mobgap.gsd.evaluation import assign_error_metrics

parameters = [
    "cadence_spm",
    "duration_s",
    "n_steps",
    "n_turns",
    "stride_duration_s",
    "stride_length_m",
    "walking_speed_mps",
]
gs_errors = assign_error_metrics(gs_matches, parameters, zero_division_hint=np.nan)

# %%
# As result, we retrieve a dataframe with two column levels containing the specified DMOs of interest together with
# their reference and detected values and the corresponding error metrics based on the discrepancy between
# the detected and reference values.
print(gs_errors)

# %%
# Aggregate Error Metrics
# -------------------------
# Finally, the estimated error metrics can be aggregated over all gait sequences.
# For this purpose, different aggregation functions can be applied to the error metrics, ranging from simple
# aggregations like the mean or standard deviation to more complex functions like the limits of agreement or
# 5th and 95th percentiles.
# Which aggregation function to apply for which parameters and errors can be configured using the helper function
# :func:`~mobgap.gsd.evaluation.get_aggregator`. It allows to specify a list of aggregation functions,
# parameters for which they should be applied, and for which type of error they should be calculated.
# Likewise, they can also be applied to the detected and reference values directly if needed.

from mobgap.gsd.evaluation import get_aggregator

aggregations = {
    **get_aggregator(aggregate=["mean", "std"], metric=["n_turns", "n_steps"], origin="detected"),
    **get_aggregator(
        aggregate=["loa", "mdc", "quantiles"], metric=["stride_duration_s", "stride_length_m"], origin="rel_error"
    ),
}

# %%
# As result, a dictionary is returned in the correct format required for further processing.
# Different aggregations can be applied to different parameters and errors by collecting them in a single dictionary
# using the unpacking operator `**`.

print(aggregations)

# %%
# Alternatively, a standard set of aggregations can be used by calling the
# :func:`~mobgap.gsd.evaluation.get_default_aggregator` function.

from mobgap.gsd.evaluation import get_default_aggregator

# TODO: which parameters are included in the default aggregator?
aggregations_default = get_default_aggregator()
print(aggregations_default)

# %%
# The aggregator dictionary can now be passed to the :func:`~mobgap.gsd.evaluation.apply_aggregations` function
# together with the error metrics dataframe to calculate the desired aggregations.

from mobgap.gsd.evaluation import apply_aggregations

agg_results = apply_aggregations(gs_errors, aggregations_default)

# %%
# The result is a dataframe containing the aggregated error metrics for each parameter and error type accumulated
# over all gait sequences in the provided data.
print(agg_results)
