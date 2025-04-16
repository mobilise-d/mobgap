r"""
.. _gsd_evaluation_parameter:

Evaluation of final walking bout level DMOs
================================================

This example shows how to evaluate the performance of parameters on a walking bout (WB) level by comparing against a reference.
On this level, we usually need to deal with the issue that the WB identified by the algorithm pipeline might not match
the reference WBs.
This makes comparing the parameters within them difficult.
In general, two approaches can be taken here [1]_:

1. First aggregate the WB-level parameters of both systems to a common level (e.g. per trial, per day, per hour, ...)
   and then compare the aggregated values.
2. Identify the subset of WBs that match between the two systems and compare the parameters only within these WBs.

In the following example we will show both approaches.

But first some general setup.

.. [1] Kirk, C., Küderle, A., Micó-Amigo, M.E. et al. Mobilise-D insights to estimate real-world walking speed in
       multiple conditions with a wearable device. Sci Rep 14, 1754 (2024). https://doi.org/10.1038/s41598-024-51766-5
"""

# %%
# Loading some example data
# -------------------------
# We simply load some example DMO data and their reference that we provide with the package.
# Usually, the "detected" data would be the output of your algorithm pipeline and the "reference" data would be the
# ground truth.
#
# .. note :: This data is randomly generated and not physiologically meaningful. However, it has the same structure as
#    any other typical input data for this evaluation.
from pprint import pprint

import numpy as np
import pandas as pd
from mobgap import PROJECT_ROOT

DATA_PATH = PROJECT_ROOT / "example_data/dmo_data/dummy_dmo_data"

detected_dmo = pd.read_csv(DATA_PATH / "detected_dmo_data.csv").set_index(
    ["visit_type", "participant_id", "measurement_date", "wb_id"]
)

reference_dmo = pd.read_csv(DATA_PATH / "reference_dmo_data.csv").set_index(
    ["visit_type", "participant_id", "measurement_date", "wb_id"]
)

# %%
# In both dataframes each row represents one WB with all of its parameters.
# The index contains multiple levels, including the visit type, participant_id, measurement day, and WB id,
# The start and end index of each WB in samples relative to the start of the respective recording is
# contained in the columns `start` and `end`.
detected_dmo

# %%
reference_dmo

# %%
# Approach 1: Aggregate then compare
# ----------------------------------
# First, we combine the detected and reference data,
# which can easily be done as both dataframes have the same index levels.
# To sustain the information about the origin of the data, we add a column level assigning `"detected"`
# and `"reference"` to the respective dmos.
# Furthermore, we rearrange the columns to have the DMO metrics as the first level of the column index.
combined_dmos = (
    pd.concat(
        [detected_dmo, reference_dmo], keys=["detected", "reference"], axis=1
    )
    .reorder_levels((1, 0), axis=1)
    .sort_index(axis=1)
)
combined_dmos

# %%
# This provides us with a dataframe containing the detected and reference values for all detected and reference WBs.
# Some entries are NaN, as the number of WBs in the detected and reference data might differ.
# The single rows in this dataframe should not be compared directly,
# as the same WB ids from a detected and a reference WB might not actually belong to the same WB.
# Therefore, we need to aggregate the DMO data based on an index level of choice, e.g., per day, to retrieve meaningful
# and interpretable results. This can for instance be done by grouping the data and averaging over the groups.
# If required, apart from simple groupwise averaging, other aggregation functions
# (e.g., moving averages or averaging over a span of several days) can be applied.
# As long as the same aggregation method is applied to both the detected and reference data,
# further processing can be done in the same way as shown below.
#
# .. note:: In case of missing data, applying `dropna()` to the resulting dataframe might be helpful to remove all
#           groups were either detected or reference data is missing.
daily_matches = (
    combined_dmos.groupby(
        level=["visit_type", "participant_id", "measurement_date"]
    )
    .mean()
    .dropna()
)
daily_matches.T

# %%
# The resulting dataframe contains the average detected and reference values for each DMO
# per visit type, participant, and day.
# This in conveniently provided as a multiindex column, so that selecting a single DMO, yields a DataFrame with the
# detected and reference values.
daily_matches["cadence_spm"]

# %%
# In our example data, we only have data from a single day, so the aggregated result only has one row.
# Normally, you would have multiple rows, one for each group of WBs.
# From here on, further processing to retrieve the aggregated error metrics is identical to the further processing when
# following approach 2, and is shown below.
#
# But let's first show how to calculate the error metrics on a WB-by-WB basis.

# %%
# Approach 2: Match then compare
# ------------------------------
# As the first step we need to indentify WBs that match between the detected and reference data.
# As it is unlikely that the WBs are exactly the same, we need to define a threshold for the overlap between the WBs
# to consider them as a match.
# This matching can be done using the :func:`~mobgap.pipeline.evaluation.categorize_intervals` function.
# It classifies every WB in the data either as true positive (TP), false positive (FP), or false negative
# (TP).
# In case our data has only WBs from a single recording, we could directly provide the detected and reference data
# to the function.
#
# However, in most cases data would contain WBs from multiple recordings, trials, and participants, ... .
# In our case, we actually only have WBs from a single recording, but we will still show the approach assuming that
# the data is more complex.
#
# To avoid, that WBs from different recordings are matched (as the matching is just performed based on the start/end
# index), we need to group the data by the relevant index levels first and apply the matching function to each group.
# This can be done using the :func:`~mobgap.utils.array_handling.create_multi_groupby` helper function.
from mobgap.utils.df_operations import create_multi_groupby

per_trial_participant_day_grouper = create_multi_groupby(
    detected_dmo,
    reference_dmo,
    groupby=["visit_type", "participant_id", "measurement_date"],
)

# %%
# This provides us with a groupby-object that is similar to the normal pandas groupby-object that can be created from a
# single dataframe.
# The ``MultiGroupBy`` object allows us to apply a function to each group across all dataframes.
#
# Here we apply :func:`~mobgap.pipeline.evaluation.categorize_intervals` with a threshold of 0.8 to each group.
# The `overlap_threshold` parameter defines the minimum overlap between the detected and reference WBs to be
# considered a match.
# It can be chosen according to your needs, whereby a value closer to 0.5 will yield more matches
# than a value closer to 1.

from mobgap.pipeline.evaluation import categorize_intervals

wb_tp_fp_fn = per_trial_participant_day_grouper.apply(
    lambda det, ref: categorize_intervals(
        gsd_list_detected=det,
        gsd_list_reference=ref,
        overlap_threshold=0.8,
        multiindex_warning=False,
    )
)
wb_tp_fp_fn

# %%
# We can see that the function returns a dataframe with the same index as the input dataframes and each WB is classified
# as TP, FP, or FN.
# For the TP WBs, the corresponding reference WB is assigned.
# For the comparison we want to perform here, only the matching WBs, i.e., the TPs, are of interest.
# If you are interested in the FPs or FNs, have a look at the general :ref:`GSD evaluation example <gsd_evaluation>`.
#
# Based on the positive matches, we can now extract the DMO data from detected and reference data that is to be
# compared.
# To make extracting all the TP WBs a little easier, we can use the
# :func:`~mobgap.pipeline.evaluation.get_matching_intervals` function.
from mobgap.pipeline.evaluation import get_matching_intervals

wb_matches = get_matching_intervals(
    metrics_detected=detected_dmo,
    metrics_reference=reference_dmo,
    matches=wb_tp_fp_fn,
)
wb_matches.T

# %%
# The returned dataframe contains the detected and reference values for all DMOs of the matched WBs.
# This in conveniently provided as a multiindex column, so that selecting a single DMO, yields a DataFrame with the
# detected and reference values.
wb_matches["cadence_spm"]

# %%
# From here on, the aggregated DMOs (when following approach 1) or matched WBs (when following approach 2)
# can be compared with the same methods to calculate error metrics. For the sake of simplicity, we will show the
# calculation of error metrics for the matched WBs `wb_matches` (approach 2) here.
# However, the input can also simply be replaced by the aggregated DMO dataframe ``
#
# Estimate Errors in DMO data
# ---------------------------
# The DMO data can now be compared day by day (approach 1) or WB by WB (approach 2).
# We want to calculate general error metrics like the error, absolute error, relative error, and absolute relative error
# for each day (WB) and DMO.
# This can be done using the generic the :func:`~mobgap.utils.df_operations.apply_transformations` helper that allows
# us to apply any list of transformation functions (transformation function -> WB in Series with same length out).
# It further allows us to declaratively define which transformation/error should be applied
# to which columns (i.e. which DMOs).
#
# A simple definition of error metrics would look like this:
# As input, it receives the matching DMO data and a list of transformations that should be applied to the data.
# A transformation is characterized as a function that takes some subset of the input dataframe,
# performs some operation on it, and returns a series with the same length as the input as output.
# Calculating the differences between two sets of values, e.g., between detected and reference values,
# is a common type of transformation that is applied to evaluate the performance of the DMO estimation.
# For this purpose, the transformations are defined as aa list of tuples containing the DMO of interest
# as the first element and the error functions applied to the detected and reference values as the second element.
# This way, you can also define custom error functions and pass them as transformations.
# Note that the columns of the detected and reference values are expected to be named `detected` and `reference`
# per default.
# For the standard error metrics (error, absolute error, relative error, absolute relative),
# the :func:`~mobgap.pipeline.evaluation.get_default_error_transformations` returns the correct transformations.
from mobgap.pipeline.evaluation import ErrorTransformFuncs as E

custom_errors = [
    ("cadence_spm", [E.abs_error, E.rel_error]),
    ("duration_s", [E.error]),
    ("n_turns", [E.rel_error]),
]

# %%
# This definition should be relatively self-explanatory.
#
# We can now apply these transformations to the DMO data using the
# :func:`~mobgap.utils.df_operations.apply_transformations`.
# Note, that there is no need to group the dataframe again, as all the transformations are applied row-wise to the
# entire dataframe.
from mobgap.utils.df_operations import apply_transformations

custom_wb_errors = apply_transformations(wb_matches, custom_errors)
custom_wb_errors.T


# %%
# We can also modify the error metrics or provide custom error functions.
# We will show three options here.
#
# 1. Use a usual error metric, but change some input parameters, and have the output under a new name.
#    For this case, we just define a new function wrapping the old one.
#    For example, we might want to suppress the warning that is raised when a zero division occurs in the relative error.
#    As we saw above, this warning is raised for the `n_turns` parameter.
def rel_error_without_warning(x):
    return E.rel_error(x, zero_division_hint=np.nan)


# %%
# 2. When we want to keep the same name for the function, we could just overwrite the old function. But,
#    to avoid accidentally messing up other code, that uses the function, we can also use a lambda function and manually
#    set the name of the function.
#    As a result, we supress the warning as above, but keep the function name for the aggregation.
rel_error_as_lambda = lambda x: E.rel_error(x, zero_division_hint=np.nan)
rel_error_as_lambda.__name__ = "rel_error"


# %%
# 3. We can also define a completely new error function.
#    The Dataframe we get as input here, contains the columns `detected` and `reference` with the detected and reference
#    values for the DMO of interest.
#    For this example here, we will create a nonsensical ``scaled_error`` function that scales the error by a factor
#    of 2.
#
# .. note::
#    If you want to introduce custom, more complex transformation functions, you can also define them as
#    :class:`~mobgap.utils.df_operations.CustomOperation` as shown for aggregations in the "Aggregation" section.
def scaled_error(x):
    return 2 * (x["detected"] - x["reference"])


# %%
# Our custom functions can now be used in the transformations list and freely combined with other error metrics.
#
# Also, keep in mind, that the definition is "just" Python, so we can use things like list comprehensions to generate
# the list of transformations as shown below.
custom_errors = [
    ("cadence_spm", [E.error, scaled_error]),
    ("duration_s", [E.error]),
    ("n_turns", [rel_error_without_warning, rel_error_as_lambda]),
    *(
        (m, [E.abs_error, E.rel_error])
        for m in ["stride_duration_s", "stride_length_m"]
    ),
]

custom_wb_errors = apply_transformations(wb_matches, custom_errors)
custom_wb_errors.T

# %%
# As expected, the resulting dataframe contains the error metrics for the specified DMOs and could now be further
# processed, e.g., by aggregating the results.
#
# As an alternative to defining a custom error definition, we provide a "default" error definition that can be used to
# calculate the standard error metrics for the common DMOs.
# In most cases, this is a good starting point for the evaluation of the DMOs.
from mobgap.pipeline.evaluation import get_default_error_transformations

default_errors = get_default_error_transformations()

pprint(default_errors)

# %%
# While the visualization here is a little ugly, we can see that the default error transformation attempts to calculate
# the error, the relative error, the absolute error, and the absolute relative error for all the core DMOs.
#
# We can apply it as before.
wb_errors = apply_transformations(wb_matches, default_errors)
wb_errors.T


# %%
# Before we now aggregate the results, we can also combine the error metrics with the reference and detected values
# to have all the information in one dataframe.
wb_matches_with_errors = pd.concat([wb_matches, wb_errors], axis=1)
wb_matches_with_errors.T

# %%
# Aggregate Results
# -----------------
# Finally, the estimated DMO measures and their errors can be aggregated over all WBs (approach 2)
# or all days (approach 1).
# For this purpose, different aggregation functions can be applied to the error metrics, ranging from simple, built-in
# aggregations like the mean or standard deviation to more complex functions like the limits of agreement or
# 5th and 95th percentiles.
# This can be done using the :func:`~mobgap.utils.df_operations.apply_aggregations` function.
# It operates similarly to the :func:`~mobgap.utils.df_operations.apply_transformations` function used above by taking
# the error metrics dataframe and a list of aggregations as input.
# In contrast to the transformations, an aggregation performed over a subset of dataframe columns
# is expected to return a single value or a tuple of values stored in one cell of the resulting dataframe.
# There are two ways to define aggregations:
#
# 1. As a tuple in the format ``(<identifier>, <aggregation>)``.
#    In this case, the operation is performed based on exactly one column from the input df.
#    Therefore, ``<identifier>`` can either be a string representing the name of the column to evaluate (for data with
#    single-level columns), or a tuple of strings uniquely identifying the column to evaluate in case of multi-index
#    columns.
#    In our example, the identifier is a tuple ``(<metric>, <origin>)``, where ``<metric>`` is the metric column to
#    evaluate, ``<origin>`` is the specific column from which data should be utilized (here, it would be either
#    ``detected``, ``reference``, or one of the error columns).
#
#    ``<aggregation>`` is the function or the list of functions to apply.
#    The output dataframe will have a multilevel column with ``metric`` as the first level and ``origin`` as the second
#    level.
#    A valid aggregations list for all of our DMOs would consequently look like this:
metrics = [
    "cadence_spm",
    "duration_s",
    "n_steps",
    "n_turns",
    "stride_duration_s",
    "stride_length_m",
    "walking_speed_mps",
]
aggregations_simple = [
    ((m, o), ["mean", "std"])
    for m in metrics
    for o in ["detected", "reference", "error"]
]
pprint(aggregations_simple)

# %%
#
# 2. As a named tuple of Type `CustomOperation` taking three values: `identifier`, `function`, and `column_name`.
#    `identifier` is a valid loc identifier selecting one or more columns from the dataframe, `function` is the (custom)
#    aggregation function, and `column_name` is the name of the resulting column in the output dataframe (single-level
#    column if `column_name` is a string, multi-level column if `column_name` is a tuple).
#    Note, that this column name needs to include the metric name.
#    For custom aggregations, we don't infer the name from the function name, as we do for the default aggregations.
#
#    This allows for more complex aggregations that require multiple columns as input, for example, the intraclass
#    correlation coefficient (ICC) for the DMOs (see below).
#    A valid aggregation list for calculating the ICC of all DMOs would look like this:
from mobgap.pipeline.evaluation import CustomErrorAggregations as A
from mobgap.pipeline.evaluation import get_default_error_aggregations
from mobgap.utils.df_operations import CustomOperation

aggregations_custom = [
    CustomOperation(identifier=m, function=A.icc, column_name=("icc", m, "all"))
    for m in metrics
]
pprint(aggregations_custom)
# %%
# In this case, the ICC function gets the entire "sub-dataframe" obtained by the selection
# ``wb_matches_with_errors.loc[:, m]`` as shown below for ``stride_duration_s`` as example, and could then perform
# any required calculations.
# The selection could theoretically be any valid loc selection.
# So you could even select values across multiple DMOs.
sub_df = wb_matches_with_errors.loc[:, "stride_duration_s"]

# %%
# The ICC function just takes the ``detected`` and ``reference`` columns and calculates the ICC.
# Note, that the function return a tuple with the ICC value and the confidence interval.
# By default, these would be put into the same column of the aggregated dataframe.
# However, we can also split them into two columns, by providing a tuple as the column name in the CustomOperation.
A.icc(sub_df)

# %%
# Let's actually do that and split the ICC and the confidence interval into two columns.
aggregations_custom = [
    CustomOperation(
        identifier=m,
        function=A.icc,
        column_name=[("icc", m, "all"), ("icc_ci", m, "all")],
    )
    for m in metrics
]

# %%
# Within one aggregation list, both types of aggregations can be combined
# as long as the resulting output dataframes can be concatenated, i.e. have the same number of column levels.
# Then, the :func:`~mobgap.utils.df_operations.apply_aggregations` function can be called.
# This returns a pandas Series with the aggregated values for each metric and origin.
# For better readability, we sort and format the resulting dataframe.
from mobgap.utils.df_operations import apply_aggregations

aggregations = aggregations_simple + aggregations_custom
agg_results = (
    apply_aggregations(wb_matches_with_errors, aggregations)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)
agg_results

# %%
# If you simply want to apply a standard set of aggregations to the error metrics, you can use the
# :func:`~mobgap.pipeline.evaluation.get_default_error_aggregations` function, resulting in the following list:

aggregations_default = get_default_error_aggregations()
pprint(aggregations_default)

# %%
# If you want to include further aggregations next to the default ones, you can also append them to this list.
aggregations_default_extended = aggregations_default + [
    *(((m, o), ["std"]) for m in metrics for o in ["detected", "reference"])
]

# %%
# This list of standard aggregations can then also be passed to the
# :func:`~mobgap.utils.df_operations.apply_aggregations` function.
default_agg_results = (
    apply_aggregations(wb_matches_with_errors, aggregations_default_extended)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)
default_agg_results

# %%
# .. note::
#   If you want to modify the default arguments of the aggregation functions, e.g. to change the calculated quantiles,
#   you can either define custom aggregation functions or adapt the default functions as shown for the transformation
#   functions above.
