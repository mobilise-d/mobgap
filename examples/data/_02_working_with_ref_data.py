"""
Working with reference data
===========================

Often you want to test an algorithmic stepo in isolation or validate the output of an algorithm.
In both cases, you need reference data - either as input or as a comparison.

As explained in the `data example <data_loading_example>`_, reference data that is stored in .mat files can be easily
loaded using the existing tooling.

In this example, we will go into more detail about common patterns of using this reference data.

Ref Data as input on a GS level
-------------------------------
Most algorithms (after the GS detection) expect the data of only a single GS.
If you want to test such an algorithm, you need to use the GS/WB information of the reference data to cut the data
accordingly.
Further, you might also want to get the reference information belonging to the GS/WB.

This can be achieved using the :func:`~gaitlink.pipeline.GsIterator`
(or the :func:`~gaitlink.pipeline.iter_gs` function).

But first, we need to load some example data.
"""
import pandas as pd

from gaitlink.data import LabExampleDataset

dataset = LabExampleDataset(reference_system="INDIP")
datapoint = dataset.get_subset(cohort="HA", participant_id="001", test="Test11", trial="Trial1")
data = datapoint.data["LowerBack"]
data

# %%
# Then we load the reference data.
# There are two versions availabele:
# One version where all values are provided relative to the start of the recording and one version where the values are
# provided relative to the start of the respective GS/WB.
# As we want to use the reference data as input to an algorithm on a GS level, we use the version relative to the start
# of the GS/WB.
#
# .. note:: The start and end values of reference WB are of course still relative to the start of the recording.
ref_data = datapoint.reference_parameters_
ref_walking_bouts = ref_data.walking_bouts

# %%
# When we look at for example the reference initial contacts, we can see that they are provided relative to the start
# of the GS/WB.
# Further, the first index level is the GS/WB id.
ref_ics = ref_data.initial_contacts
ref_ics
# %%
# Now we can use the :func:`~gaitlink.pipeline.GsIterator` to iterate over the data.
# Check out the `gs_iterator example <gs_iterator_example>`_ for more information.
# But basically, we need a data-class to define the results we expect, and then we can simply iterate over the data.
from dataclasses import make_dataclass

from gaitlink.pipeline import GsIterator

# TODO: Update once we make the defaults for the iterator more convenient for these standard usecases

expected_results = make_dataclass("expected_results", [("initial_contacts", pd.Series)])
gs_iterator = GsIterator(expected_results)

# %%
# The iterator provides us the cut data and the id of the respective GS/WB per iteration.
# The latter can be used to index other aspects of the reference data.
for (wb, data_per_wb), result in gs_iterator.iterate(data, ref_walking_bouts):
    print("GS/WB id: ", wb.wb_id)
    print("Expected N-samples in wb: ", ref_walking_bouts.loc[wb.wb_id].end - ref_walking_bouts.loc[wb.wb_id].start)
    print("N-samples in wb: ", len(data_per_wb))

    # We can use the wb.wb_id to get the reference initial contacts that belong to this GS/WB
    ics_per_wb = ref_ics.loc[wb.wb_id]
    # These could be used in some algorithm.
    # Here we will just store them in the results.
    result.initial_contacts = ics_per_wb

# %%
# Inspecting the results, we can see that the initial contacts are stored as expected, and we get the correct dataframe
# per GS (compare to the output from above).
# The results are list of with one entry per iteration.
gs_iterator.initial_contacts_

# %%
# For better comparison, let's just have a look at the first entry.
gs_iterator.initial_contacts_.iloc[0]
