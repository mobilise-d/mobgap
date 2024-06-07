r"""
.. _gsd_evaluation_parameter:

GSD Evaluation based on DMO outcomes
====================================

This example shows how to evaluate the performance of the gait sequence detection on gait sequence level.
This is done by comparing the DMOs estimated based on the detected gait sequences with the DMOs estimated based on the
ground truth gait sequences.
"""

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
from pprint import pprint

import numpy as np
import pandas as pd

from mobgap import PACKAGE_ROOT

DATA_PATH = PACKAGE_ROOT.parent / "example_data/dmo_data/dummy_dmo_data"

detected_dmo = pd.read_csv(DATA_PATH / "detected_dmo_data.csv").set_index(
    ["visit_type", "participant_id", "measurement_date", "wb_id"]
)

reference_dmo = pd.read_csv(DATA_PATH / "reference_dmo_data.csv").set_index(
    ["visit_type", "participant_id", "measurement_date", "wb_id"]
)

# %% The DMO data is a dataframe with one row per gait sequence.
# The index contains multiple levels, including the visit type, participant_id, measurement day, and gait sequence id,
# and the corresponding estimated DMOs.
# Furthermore, the start and end index of each gait sequence in samples is contained in the columns `start` and `end`.
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
    detected_dmo, reference_dmo, groupby=["visit_type", "participant_id", "measurement_date"]
)

pprint(per_trial_participant_day_grouper)

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

from mobgap.gsd.evaluation import CustomOperation, categorize_matches_with_min_overlap, icc

gs_tp_fp_fn = create_multi_groupby(
    detected_dmo, reference_dmo, groupby=["visit_type", "participant_id", "measurement_date"]
).apply(
    lambda det, ref: categorize_matches_with_min_overlap(
        gsd_list_detected=det, gsd_list_reference=ref, overlap_threshold=0.8, multiindex_warning=False
    )
)
gs_tp_fp_fn


# %%
# Based on the positive matches,
# we can now extract the DMO data from detected and reference data that is to be compared.
# To combine this data, we use the :func:`~mobgap.gsd.evaluation.get_matching_gs` function.

from mobgap.gsd.evaluation import get_matching_gs

gs_matches = get_matching_gs(metrics_detected=detected_dmo, metrics_reference=reference_dmo, matches=gs_tp_fp_fn)
gs_matches

# %%
# Estimate Errors in DMO data
# ----------------------------
# The DMO data can now be compared gait sequence by gait sequence.
# To estimate discrepancies between the detected and reference DMOs,
# the :func:`~mobgap.gsd.evaluation.apply_transformations` can be used.
# As input, it receives the matching DMO data and a list of transformations that should be applied to the data.
# A transformation is characterized as a function that takes some subset of the input dataframe,
# performs some operation on it, and returns a series as output.
# The transformations are a list of tuples containing the DMO of interest as the first element and the error functions
# applied to the detected and reference values as the second element.
# This way, you can also define custom error functions and pass them as transformations.
# Note that the columns of the detected and reference values are expected to be named `detected` and `reference`
# per default.
# For the standard error metrics (error, absolute error, relative error, absolute relative),
# the :func:`~mobgap.gsd.evaluation.get_default_error_transformations` returns the correct transformations.
from mobgap.gsd.evaluation import apply_transformations, get_default_error_transformations

default_errors = get_default_error_transformations()
pprint(default_errors)

# %%
# The specified transformations are then applied to the DMO data
# using the :func:`~mobgap.gsd.evaluation.apply_transformations`.
# Within this function, to every of the transformation functions listed for a DMO in the `default_errors` list,
# the columns of the `gs_matches` input dataframe containing data from this DMO are passed as input.
# When following the steps in this example, this input data contains a column `detected` and a column `reference`,
# which are then automatically selected by the default error functions. If your analysis should be based on different
# columns, you can adapt the error functions as described in the next section.
gs_errors = apply_transformations(gs_matches, default_errors)
gs_errors

# %%
# Modify Transformation Functions
# -------------------------------
# For some DMOs, it might occur that the reference value is 0, which would lead to a division by zero when calculating
# the relative error. In this example, this happens for the `n_turns` parameter.
# Per default, the function raises a warning when zero division occurs and sets the relative error to NaN.
# To suppress the warning, the default error function arguments can be modified by defining adapted functions
# setting the `zero_division_hint` parameter to `np.nan`.
# We set the name of the adapted functions to the names of the original functions,
# as the columns in the output dataframe are named after the functions.
# The transformations list then needs to be updated with the adapted functions.
# This way, no warning will be raised when zero division occurs.
# In the same manner, the default names `detected` and `reference` of the two input columns for the error functions
# can be overwritten by custom names.
# ..note:: The adapted functions are defined as lambda functions here for simplicity.
# The function argument `x` contains the detected and reference columns of the DMO of interest.
# Naming these lambda function is required for this application,
# since the function names are used as column names in the output dataframe.
# It may seem counterintuitive that the adapted functions are named the same as the original functions.
# However, for further processing including aggregations, those names are required for the default aggregations to work.
from mobgap.gsd.evaluation import abs_error, abs_rel_error, error, rel_error

rel_err_suppressed_warning = lambda x: rel_error(x, zero_division_hint=np.nan)
rel_err_suppressed_warning.__name__ = "rel_error"

abs_rel_err_suppressed_warning = lambda x: abs_rel_error(x, zero_division_hint=np.nan)
abs_rel_err_suppressed_warning.__name__ = "abs_rel_error"

metrics = [
    "cadence_spm",
    "duration_s",
    "n_steps",
    "n_turns",
    "stride_duration_s",
    "stride_length_m",
    "walking_speed_mps",
]
adapted_errors = [error, rel_err_suppressed_warning, abs_error, abs_rel_err_suppressed_warning]
error_metrics = [*((m, adapted_errors) for m in metrics)]
gs_errors_adapted = apply_transformations(gs_matches, error_metrics)
gs_errors_adapted

# %%
# The resulting dataframe contains the errors for all metrics and walking bouts.
# It can be concatenated with the reference and detected values to obtain a comprehensive overview of the DMOs.
gs_matches_with_errors = pd.concat([gs_matches, gs_errors], axis=1)
gs_matches_with_errors

# %% .. note::
# If you want to introduce custom, more complex transformation functions, you can also define them as
# `CustomOperation` as shown for aggregations in the following section.

# %%
# Aggregate Results
# -----------------
# Finally, the estimated DMO measures and their errors can be aggregated over all gait sequences.
# For this purpose, different aggregation functions can be applied to the error metrics, ranging from simple, built-in
# aggregations like the mean or standard deviation to more complex functions like the limits of agreement or
# 5th and 95th percentiles.
# This can be done using the :func:`~mobgap.gsd.evaluation.apply_aggregations` function.
# It operates similarly to the :func:`~mobgap.gsd.evaluation.apply_transformations` function used above
# by taking the error metrics dataframe and a list of aggregations as input.
# In contrast to the transformations, an aggregation performed over a subset of dataframe columns
# returns a single value or a tuple of values stored in one cell of the resulting dataframe.
# There are two ways to define aggregations:
#
# 1. As a tuple in the format `(<identifier>, <aggregation>)`.
#    In this case, the operation is performed based on exactly one column from the input df.
#    Therefore, <identifier> can either be a string representing the name of the column to evaluate
#    (for data with single-level columns),
#    or a tuple of strings uniquely identifying the column to evaluate.
#    In our example, the identifier is a tuple (<metric>, <origin>),
#    where `<metric>` is the metric column to evaluate,
#    `<origin>` is the specific column from which data should be utilized
#    (here, it would be either `detected`, `reference`, or one of the error columns)
#    (e.g., this example, `detected`, `reference`, or `error`).
#    Furthermore, `<aggregation>` is the function or the list of functions to apply.
#    The output dataframe will have a multilevel column with `metric` as the first level and
#    `origin` as the second level.
#    A valid aggregations list for all of our DMOs would consequently look like this:

aggregations_simple = [*(((m, o), ["mean", "std"]) for m in metrics for o in ["detected", "reference", "error"])]
pprint(aggregations_simple)

# %%
#
# 2. As a named tuple of Type `CustomOperation` taking three values: `identifier`, `function`, and `column_name`.
#   `identifier` is a valid loc identifier selecting one or more columns from the dataframe,
#   `function` is the (custom) aggregation function or list of functions to apply,
#   and `column_name` is the name of the resulting column in the output dataframe
#   (single-level column if `column_name` is a string, multi-level column if `column_name` is a tuple).
#   This allows for more complex aggregations that require multiple columns as input, for example, the intraclass
#   correlation coefficient (ICC) for the DMOs.
#   A valid aggregation list for calculating the ICC of all DMOs would look like this:

aggregations_custom = [CustomOperation(identifier=m, function=icc, column_name=(m, "all")) for m in metrics]
pprint(aggregations_custom)

# %%
# Within one aggregation list, both types of aggregations can be combined
# as long as the resulting output dataframes can be concatenated, i.e. have the same number of column levels.
# Then, the :func:`~mobgap.gsd.evaluation.apply_aggregations` function can be called.
# For better readability, the index of the aggregation result dataframe is named and sorted.
from mobgap.gsd.evaluation import apply_aggregations

aggregations = aggregations_simple + aggregations_custom
agg_results = apply_aggregations(gs_matches_with_errors, aggregations)
agg_results = agg_results.rename_axis(index=["aggregation", "metric", "origin"])
agg_results = agg_results.reorder_levels(["metric", "origin", "aggregation"]).sort_index(level=0)
pd.DataFrame(agg_results)

# %%
# If you simply want to apply a standard set of aggregations to the error metrics, you can use the
# :func:`~mobgap.gsd.evaluation.get_default_aggregations` function, resulting in the following list:
from mobgap.gsd.evaluation import get_default_aggregations

aggregations_default = get_default_aggregations()
pprint(aggregations_default)

# %% If you want to include further aggregations next to the default ones, you can also append them to this list.
aggregations_default_extended = aggregations_default + [
    *(((m, o), ["std"]) for m in metrics for o in ["detected", "reference"])
]

# %%
# This list of standard aggregations can then also be passed to the :func:`~mobgap.gsd.evaluation.apply_aggregations`
# function.
default_agg_results = apply_aggregations(gs_matches_with_errors, aggregations_default_extended)
default_agg_results = default_agg_results.rename_axis(index=["aggregation", "metric", "origin"])
default_agg_results = default_agg_results.reorder_levels(["metric", "origin", "aggregation"]).sort_index(level=0)
pd.DataFrame(default_agg_results)

# %%
# .. note::
#   If you want to modify the default arguments of the aggregation functions, e.g. to change the calculated quantiles,
#   you can either define custom aggregation functions or adapt the default functions as shown for the transformation
#   functions in the section `Modify Transformation Functions` above.
