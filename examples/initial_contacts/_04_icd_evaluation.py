r"""
.. _icd_evaluation:

ICD Evaluation
==============

This example shows how to apply evaluation algorithms to ICD and thus how to rate the performance of an ICD algorithm.
"""

import pandas as pd

# %%
# Import useful modules and packages
from mobgap.data import LabExampleDataset
from mobgap.initial_contacts import IcdIonescu
from mobgap.pipeline import GsIterator
from mobgap.utils.conversions import to_body_frame

# %%
# Loading some example data
# -------------------------
# First, we load example data and apply the ICD Ionescu algorithm to it.
# However, you can use any other ICD algorithm as well.
# To have a reference to compare the results to, we also load the corresponding ground truth data.
# These steps are explained in more detail in the :ref:`ICD Ionescu example <icd_ionescu>`.


def load_data():
    """Load example data and extract a single trial for demonstration purposes."""
    example_data = LabExampleDataset(
        reference_system="INDIP", reference_para_level="wb"
    )
    single_test = example_data.get_subset(
        cohort="HA", participant_id="001", test="Test11", trial="Trial1"
    )
    return single_test


def calculate_icd_ionescu_output(single_test_data):
    """Calculate the ICD Ionescu output for one sensor from the test data."""
    imu_data = to_body_frame(single_test_data.data_ss)
    sampling_rate_hz = single_test_data.sampling_rate_hz
    reference_wbs = single_test_data.reference_parameters_.wb_list

    iterator = GsIterator()
    for (gs, data), result in iterator.iterate(imu_data, reference_wbs):
        result.ic_list = (
            IcdIonescu()
            .detect(data, sampling_rate_hz=sampling_rate_hz)
            .ic_list_
        )

    det_ics = iterator.results_.ic_list
    return det_ics


def load_reference(single_test_data):
    """Load the reference initial contacts from the test data."""
    ref_ics = single_test_data.reference_parameters_.ic_list
    return ref_ics


wb_data = load_data()
detected_ics = calculate_icd_ionescu_output(wb_data)
reference_ics = load_reference(wb_data)

# %%
# As you can see our detected initial contacts and reference initial contacts are multiindexed dataframes.
# The first level of the multiindex is the walking bout id and the second level is the index of the initial contact
# within the walking bout.
detected_ics

# %%
reference_ics

# %%
# Matching ICs between detected and reference lists
# -------------------------------------------------
# Let's quantify how the algorithm output compares to the reference labels.
# To gain a detailed insight into the performance of the algorithm, we can look into the individual matches between the
# detected and reference initial contacts.
# To do this, we use the :func:`~mobgap.initial_contacts.evaluation.categorize_ic_list` function to classify each detected initial
# contact as a true positive, false positive, or false negative.
# We can then use these results to calculate a range of higher-level performance metrics.
#
# Note, that we want to only match initial contacts within the same walking bout.
# If we would simply pass the detected and reference initial contacts to the matching function, it would match all ICs
# independent of the walking bout, as it ignores the multiindex.
# We will have a look at how this looks like below, and when we might want to use it, but for now, let's perform the
# matching within the walking bouts.
#
# For this, we need to group the detected and reference initial contacts by the walking bout id.
# This can be done using the :func:`~mobgap.utils.array_handling.create_multi_groupby` helper function.
from mobgap.utils.df_operations import create_multi_groupby

per_wb_grouper = create_multi_groupby(
    detected_ics, reference_ics, groupby="wb_id"
)

# %%
# The provides us with a groupby object that is similar to the normal pandas groupby object that can be created from a
# single dataframe.
# The ``MultiGroupBy`` object allows us to apply a function to each group across all dataframes.
# I.e. the function will get the detected and reference initial contacts for each walking bout and then can perform
# some operation on them.
#
# In our case we want to apply the :func:`~mobgap.initial_contacts.evaluation.categorize_ic_list` function to each walking bout.
# This function will then return a dataframe with the matches given a certain tolerance.
#
# We don't assume that initial contacts are detected at perfectly the exact same time in both systems.
# Hence, we allow for a certain deviation in the matching process.
from mobgap.utils.conversions import as_samples

tolerance_s = 0.2
tolerance_samples = as_samples(tolerance_s, wb_data.sampling_rate_hz)
tolerance_samples

# %%
# Now we can apply the matching function to each walking bout.
# Note, that our matches retain the multiindex and provide matches for each walking bout separately.
# The dataframe has 3 columns, containing the index value of the detected ic, the index value of matched reference ic,
# and the match type.
# The two index columns contain tuples in our case, as they stem from the original multiindex that we provided.
# So each of the tuples has the form ``(wb_id, ic_id)``.
from mobgap.initial_contacts.evaluation import categorize_ic_list

matches_per_wb = create_multi_groupby(
    detected_ics, reference_ics, groupby="wb_id"
).apply(
    lambda df1, df2: categorize_ic_list(
        ic_list_detected=df1,
        ic_list_reference=df2,
        tolerance_samples=tolerance_samples,
        multiindex_warning=False,
    )
)
matches_per_wb

# %%
# Instead of matching the initial contacts within the same walking bout, we could also match all initial contacts
# independent of the walking bout.
# This can be done by simply passing the detected and reference initial contacts directly to the matching function.
# This can be useful if the walking bouts between the two compared systems are not identical or the multiindex has
# other columns that should not be taken into account for the matching.
matched_all = categorize_ic_list(
    ic_list_detected=detected_ics,
    ic_list_reference=reference_ics,
    tolerance_samples=tolerance_samples,
)

matched_all
# %%
# Note, that this did not really make a difference in our case, as the individual WBs are identical between the two
# systems and far enough apart so that matches between different WBs are not possible.
# But in general, this can be a typical "foot-gun" for users, as they might not be aware of the fact that the multiindex
# is ignored in the matching process.
# Hence, as you can see above, a warning is raised if you pass a multiindex to the matching function.
# This can be silenced by setting the ``multiindex_warning`` parameter to ``False``.
#
# As in our case we would recommend to match the ICs per walking bout, we will continue with the matches per walking
# bout and ignore ``matches_all`` for the rest of this example.
#
# Calculating performance metrics
# -------------------------------
# From these ``matches_per_wb``, a range of higher-level performance metrics (including the total number of true
# positives, false positives, and false negatives, as well as precision, recall, and F1-score) can be calculated.
# For this purpose, we can use the :func:`~mobgap.initial_contacts.evaluation.calculate_matched_icd_performance_metrics` function.
# It returns a dictionary containing all metrics for the specified detected and reference initial contact lists.
#
# We can again decide, if we want to calculate these metrics across all walking bouts or for each walking bout
# separately.
# We will quickly show both approaches below.
#
# Across all walking bouts:
from mobgap.initial_contacts.evaluation import (
    calculate_matched_icd_performance_metrics,
)

metrics_all = calculate_matched_icd_performance_metrics(matches_per_wb)

pd.Series(metrics_all)

# %%
# Per Wb:
#
# For this we can use the normal pandas groupby to calculate the metrics for each walking bout separately.
metrics_per_wb = matches_per_wb.groupby(level="wb_id").apply(
    lambda df_: pd.Series(calculate_matched_icd_performance_metrics(df_))
)

metrics_per_wb

# %%
# Which of the two approaches makes more sense depends on the use case and what your multiindex represents.
