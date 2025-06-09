r"""
.. _cad_evaluation:

Cadence Evaluation
==================

This example shows how to apply evaluation algorithms to cadence results
and thus how to rate the performance of a cadence calculation algorithm.
"""

from pprint import pprint

import numpy as np
import pandas as pd
from mobgap.data import LabExampleDataset

# %%
# Example Data
# ------------
# We load example data from the lab dataset.
# We will use a longer trial from an "MS" participant for this example.
# Additionally, we will load the reference cadence values to compare the results and evaluate the performance of the
# cadence calculation algorithm.
# To calculate the cadence, we will then use the initial contacts measured from the reference system.


def load_data():
    """Load example data from the lab dataset."""
    lab_example_data = LabExampleDataset(reference_system="INDIP")
    long_trial = lab_example_data.get_subset(
        cohort="MS", participant_id="001", test="Test11", trial="Trial1"
    )
    sampling_rate_hz = long_trial.sampling_rate_hz
    return long_trial, sampling_rate_hz


def load_reference(data):
    """Load reference from the INDIP reference system."""
    reference_gs = data.reference_parameters_.wb_list
    reference_ic = data.reference_parameters_relative_to_wb_.ic_list
    return reference_gs, reference_ic


test_data, sampling_rate_hz = load_data()
reference_gs, reference_ic = load_reference(test_data)

# %%
# From the reference data, we can see that our example data contains several gait sequences.
# For each gait sequence, the reference system provides an average cadence value.
reference_gs[["avg_cadence_spm"]]

# %%
# Furthermore, the reference data includes a list of initial contacts for each gait sequence.
# We will use these reference initial contacts as input for the cadence calculation algorithm,
# to enable evaluating the performance of the cadence calculation in isolation.
reference_ic

# %%
# Applying the Cadence Calculation Algorithm
# ------------------------------------------
# In this example, we will use the :class:`~mobgap.cadence.CadFromIc` algorithm to calculate the cadence
# from the reference initial contacts.
from mobgap.cadence import CadFromIc

cad_from_ic = CadFromIc()


# %%
# As this algorithm is designed to work on a single gait sequence at a time, we will iterate over the gait sequences
# present in the example data and calculate the cadence for each of them.
# This is done using the :class:`~mobgap.pipeline.GsIterator` class.
# How the :class:`~mobgap.pipeline.GsIterator` class works in detail together with different application examples
# is explained in its :ref:`dedicated example <gs_iterator_example>`.
from mobgap.initial_contacts import refine_gs
from mobgap.pipeline import GsIterator
from mobgap.utils.conversions import to_body_frame

iterator = GsIterator()

for (gs, data), r in iterator.iterate(test_data.data_ss, reference_gs):
    r.ic_list = reference_ic.loc[gs.id]
    refined_gs, refined_ic_list = refine_gs(r.ic_list)
    with iterator.subregion(refined_gs) as ((_, refined_gs_data), rr):
        cad = cad_from_ic.clone().calculate(
            to_body_frame(refined_gs_data),
            initial_contacts=refined_ic_list,
            sampling_rate_hz=sampling_rate_hz,
        )
        rr.cadence_per_sec = cad.cadence_per_sec_

# %%
# The detected cadences per second for all gait sequences
# can then be accessed from the `results_` attribute of the `GsIterator` object.
cadence_result = iterator.results_.cadence_per_sec
cadence_result

# %%
# Comparison with Reference
# -------------------------
# To evaluate the performance of the cadence calculation algorithm, we can compare the calculated cadence values
# with the reference cadence values.
# For this purpose, as there are is only one reference cadence value per gait sequence, we will first average the
# calculated cadence values per gait sequence.
avg_cadence_per_gs = cadence_result.groupby("wb_id").mean()
avg_cadence_per_gs

# %%
# Next, the detected and reference cadences values are concatenated into a single DataFrame.
# As columns, a multilevel index is used that contains the type of metric (``cadence_spm``) in the first
# and the source of the data (``detected`` or ``reference``) in the second level.
reference_cadence = reference_gs[["avg_cadence_spm"]].rename(
    columns={"avg_cadence_spm": "cadence_spm"}
)
combined_cad = {"detected": avg_cadence_per_gs, "reference": reference_cadence}
combined_cad = pd.concat(
    combined_cad, axis=1, keys=combined_cad.keys()
).reorder_levels((1, 0), axis=1)
combined_cad

# %%
# The concatenated DataFrame can then be used to evaluate the performance of the cadence calculation algorithm.
# For this purpose, first,
# the errors between the detected and reference cadence values are calculated.
#
# Estimate Errors in cadence data
# +++++++++++++++++++++++++++++++
# We can calculate a variety of error metrics to evaluate the performance of the cadence calculation algorithm,
# ranging from the simple difference between the estimated and ground truth values (simply referred to as ``error``)
# to its absolute value (``absolute_error``).
# Both can also be set in relation to the reference value (``relative_error`` and ``absolute_relative_error``).
# To apply these errors, we first need to build a list specifying the error functions to be applied.
# All the above-mentioned error functions are provided
# by the :class:`~mobgap.pipeline.evaluation.ErrorTransformFuncs` class.
# Note that you can also apply custom functions instead of the predefined ones.
# For more information on how to define custom error functions,
# see the :ref:`general example on DMO evaluation <gsd_evaluation_parameter>`.
from mobgap.pipeline.evaluation import ErrorTransformFuncs as E

errors = [("cadence_spm", [E.error, E.abs_error, E.rel_error, E.abs_rel_error])]
pprint(errors)

# %%
# The error functions can be applied to the combined cadence data
# using the :func:`~mobgap.utils.df_operations.apply_transformations` function.
# The resulting DataFrame contains the error values for each gait sequence.
from mobgap.utils.df_operations import apply_transformations

cad_errors = apply_transformations(combined_cad, errors)
cad_errors.T

# %%
# Before we now aggregate the results, we can also combine the error metrics with the reference and detected values
# to have all the information in one dataframe.
combined_cad_with_errors = pd.concat([combined_cad, cad_errors], axis=1)
combined_cad_with_errors

# %%
# Aggregate Errors
# ++++++++++++++++
# Finally, the estimated errors can be aggregated to provide a summary of the performance of the cadence calculation.
# For this purpose, different aggregation functions can be applied to the error metrics, ranging from simple, built-in
# aggregations like the mean or standard deviation to more complex functions like the limits of agreement or
# 5th and 95th percentiles.
# This can be done using the :func:`~mobgap.utils.df_operations.apply_aggregations` function.
# Possible aggregations are provided by the :class:`~mobgap.pipeline.evaluation.CustomErrorAggregations` class.
# There are two ways to define such aggregations:
#
# 1. As a list of tuples in the format ``(<identifier>, <aggregation>)`` with
#    ``<identifier>`` being the key for accessing the column to evaluate,
#    and ``<aggregation>`` being the aggregation function(s) to apply.
#    A valid list of aggregations could look like this:
from mobgap.pipeline.evaluation import CustomErrorAggregations as A

aggregations_simple = [
    *(
        (("cadence_spm", origin), [np.mean, A.quantiles])
        for origin in ["detected", "reference", "abs_error", "abs_rel_error"]
    ),
    *(
        (("cadence_spm", origin), [np.mean, A.loa])
        for origin in ["error", "rel_error"]
    ),
]
pprint(aggregations_simple)

# %%
# 2. As a named tuple of Type `CustomOperation` taking three values: `identifier`, `function`, and `column_name`.
#    `identifier` is a valid loc identifier selecting one or more columns from the dataframe, `function` is the
#    aggregation function or list of functions to apply,
#    and `column_name` is the identifier of the resulting column in the output dataframe.
#    This allows for more complex aggregations that require multiple columns as input, for example, the intra-class
#    correlation coefficient (ICC).
#    A valid aggregation list for calculating the ICC of all DMOs would look like this:
from mobgap.utils.df_operations import CustomOperation

aggregations_custom = [
    CustomOperation(
        identifier="cadence_spm",
        function=A.icc,
        column_name=("icc", "cadence_spm", "all"),
    )
]
pprint(aggregations_custom)

# %%
# For more detailed information on the aggregation types and their usage,
# check out the detailed example on it in the :ref:`general example on DMO evaluation <gsd_evaluation_parameter>`.
#
# Both types of aggregations can be combined and applied in a single call to the
# :func:`~mobgap.utils.df_operations.apply_aggregations` function.
# This returns a pandas Series with the aggregated values for each aggregation function and origin
# for the metric cadence.
# For better readability, we sort and format the resulting dataframe.
from mobgap.utils.df_operations import apply_aggregations

aggregations = aggregations_simple + aggregations_custom
agg_results = (
    apply_aggregations(combined_cad_with_errors, aggregations)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)
agg_results
