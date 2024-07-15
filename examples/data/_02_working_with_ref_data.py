"""
Working with reference data
===========================

Often you want to test an algorithmic step in isolation or validate the output of an algorithm.
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

This can be achieved using the :func:`~mobgap.pipeline.GsIterator`
(or the :func:`~mobgap.pipeline.iter_gs` function).

But first, we need to load some example data.
"""

from mobgap.data import LabExampleDataset

dataset = LabExampleDataset(reference_system="INDIP")
datapoint = dataset.get_subset(
    cohort="HA", participant_id="001", test="Test11", trial="Trial1"
)
data = datapoint.data_ss
data

# %%
# Then we load the reference data.
# There are two versions available:
# One version where all values are provided relative to the start of the recording and one version where the values are
# provided relative to the start of the respective GS/WB.
# Below we can see the first version, here both the walking bouts and the initial contacts (and other parameters) are
# provided relative to the start of the recording.
ref_data = datapoint.reference_parameters_
ref_data.wb_list

# %%
ref_data.ic_list

# %%
# However, as we want to use the reference data as input to an algorithm on a GS level, we use the version that provides
# values relative to the start of the GS/WB.
#
# The start and end values of reference WB are of course still relative to the start of the recording.
ref_data_rel = datapoint.reference_parameters_relative_to_wb_
ref_walking_bouts = ref_data_rel.wb_list
ref_walking_bouts

# %%
# But the ICs time-samples are now relative to the start of the respective GS/WB.
ref_ics_rel = ref_data_rel.ic_list
ref_ics_rel.loc[0]  # First WB

# %%
ref_ics_rel.loc[1]  # Second WB

# %%
# Now we can use the :func:`~mobgap.pipeline.GsIterator` to iterate over the data.
# Check out the `gs_iterator example <gs_iterator_example>`_ for more information.
from mobgap.pipeline import GsIterator

gs_iterator = GsIterator()

# For most use-cases, the default configuration of the :class:`~mobgap.pipeline.GsIterator` should be sufficient.
# This allows you to specify the following results:
gs_iterator.data_type

# %%
# If you want to change the default behaviour, you can create a custom dataclass (check the example linked above)
#
# The iterator provides us the cut data and an object representing all information of the respective GS/WB.
# The latter can be used to index other aspects of the reference data.
for (wb, data_per_wb), result in gs_iterator.iterate(data, ref_walking_bouts):
    print("GS/WB id: ", wb.id)
    print(
        "Expected N-samples in wb: ",
        ref_walking_bouts.loc[wb.id].end - ref_walking_bouts.loc[wb.id].start,
    )
    print("N-samples in wb: ", len(data_per_wb))

    # We can use the wb.id to get the reference initial contacts that belong to this GS/WB
    ics_per_wb = ref_ics_rel.loc[wb.id]
    # These could be used in some algorithm.
    # Here we will just store them in the results.
    result.ic_list = ics_per_wb

# %%
# The iterator will also conveniently aggregate the results for us.
# You can see that the initial contacts are now stored in a single dataframe and the values are transformed back to be
# relative to the start of the recording and not the GS anymore.
gs_iterator.results_.ic_list
