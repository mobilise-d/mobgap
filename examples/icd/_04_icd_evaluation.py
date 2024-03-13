r"""
.. _icd_evaluation:

ICD Evaluation
==============

This example shows how to apply evaluation algorithms to ICD and thus how to rate the performance of an ICD algorithm.
"""
# %%
# Import useful modules and packages
from gaitlink.data import LabExampleDataset
from gaitlink.icd import IcdIonescu
from gaitlink.pipeline import GsIterator

# %%
# Loading some example data
# -------------------------
# First, we load example data and apply the ICD Ionescu algorithm to it. To have a reference to compare the results to,
# we also load the correcponding ground truth data.
# These steps are explained in more detail in the :ref:`ICD Ionescu example <icd_ionescu>`.


def load_data():
    """Load example data and extract a single trial for demonstration purposes."""
    example_data = LabExampleDataset(reference_system="INDIP", reference_para_level="wb")
    single_test = example_data.get_subset(cohort="HA", participant_id="001", test="Test11", trial="Trial1")
    return single_test


def calculate_icd_ionescu_output(single_test_data):
    """Calculate the ICD Ionescu output for one sensor from the test data."""
    imu_data = single_test_data.data["LowerBack"]
    sampling_rate_hz = single_test_data.sampling_rate_hz
    reference_wbs = single_test_data.reference_parameters_.wb_list

    iterator = GsIterator()
    for (gs, data), result in iterator.iterate(imu_data, reference_wbs):
        result.ic_list = IcdIonescu().detect(data, sampling_rate_hz=sampling_rate_hz).ic_list_

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
# Evaluation of the algorithm against a reference
# --------------------------------------------------
# Let's quantify how the algorithm output compares to the reference labels.
# On the one hand, to gain a detailed insight into the performance of the algorithm,
# we can look into the individual matches between the detected and reference initial contacts.
# To do this, we use the :func:`~gaitlink.icd.evaluation.evaluate_ic_list` function to compare the detected
# ICs to the ground truth ICs.
# The :func:`~gaitlink.icd.evaluation.evaluate_ic_list` function will "ignore" the multiindex
# by default and will potentially match initial contacts across different walking bouts.
# This can be useful if the walking bouts between the two compared systems are not identical or the multiindex has
# other columns that should not be taken into account for the matching.
#
# However, in our case, we used the WBs from the reference system to iterate over the data and apply the algorithm.
# This means we have exactly the same WBs and only want to match ICs within the same WB.
# For this we use the :func:`~gaitlink.utils.array_handling.create_multi_groupby` helper to apply the matching func to each
# WB separately.
# We also set the `multiindex_warning` parameter to `False` to suppress the warning that is otherwise raised informing
# users about this potential "foot-gun".
#
# With the `tolerance_samples` parameter, we can specify the maximum allowed deviation in samples.
# Consequently, the tolerance parameter should be chosen with respect to the sampling rate of the data.
# In this case, it is set to 20 samples, which corresponds to 200 ms.
# As our data includes multiple walking bouts and the detected initial contacts within these walking bouts,
# it has a multiindex with two levels to indicate that.

from gaitlink.icd.evaluation import evaluate_ic_list
from gaitlink.utils.array_handling import create_multi_groupby

matches = create_multi_groupby(detected_ics, reference_ics, groupby=["wb_id"]).apply(
    lambda df1, df2: evaluate_ic_list(
        ic_list_detected=df1, ic_list_reference=df2, tolerance_samples=20, multiindex_warning=False
    )
)

# %%
# The function returns a 3-column dataframe with the columns containing the index value of the detected ic,
# the index value of matched reference ic, and the match type.
# The index serves as a unique identifier for each initial contacts,
# therefore it must be unique for each ic.
# Our multiindex is simply flattened to a tuple in the matching process.
# We can see that for all ICs that have a match in the reference list, the match type is be "tp" (true positive).
# ICs that do not have a match are be mapped to a NaN and the match-type is "fp" (false
# positive). All reference initial contacts that do not have a counterpart in the detected list
# are marked as "fn" (false negative).

matches

# %%
# From these detailed results, a range of higher-level performance metrics (including the total number of
# true positives, false positives, and false negatives, as well as precision, recall, and F1-score) can be calculated.
# For this purpose,
# we can use the :func:`~gaitlink.icd.evaluation.calculate_icd_performance_metrics` function.
# It returns a dictionary containing all metrics for the specified detected and reference initial contact lists.
#
# As our example data only contains a single participant and a single recording session, we want to aggregate one set
# of performance metrics for all walking bouts. Thus, we can simply use the entire matches dataframe as input.
# However, if your data contains multiple participants or recording sessions, you might want to use the
# :func:`~gaitlink.utils.array_handling.create_multi_groupby` function again to retrieve the performance metrics for
# subsets of the data.

from gaitlink.icd.evaluation import calculate_icd_performance_metrics

metrics = calculate_icd_performance_metrics(matches)

metrics
