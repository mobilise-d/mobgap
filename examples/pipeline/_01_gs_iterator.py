"""
.. _gs_iterator_example:

Gait Sequence Iterator
======================

As part of most pipelines, we need to iterate over the gait sequences to apply all further algorithms to them
individually.
This can be a bit cumbersome, as we need to iterate over the data and aggregate the results at the same time.
Hence, we provide some helpers for that.

We provide two ways of iterating.
The first one, only handles the iteration and does not aggregate the results.
The second approach attempts to also support you in aggregating the results.

Getting Some Example Data
-------------------------

"""

import numpy as np
import pandas as pd

from mobgap.data import LabExampleDataset

lab_example_data = LabExampleDataset(reference_system="INDIP")
long_trial = lab_example_data.get_subset(cohort="MS", participant_id="001", test="Test11", trial="Trial1")
long_trial_gs = long_trial.reference_parameters_.wb_list

long_trial_gs

# %%
# Simple Functional Interface
# ---------------------------
# We provide the :func:`~mobgap.pipeline.iter_gs` function to iterate over the gait sequences.
# It simply takes the data and the gait sequence list and cuts the data accordingly to iterate over it.
# The function yields the gait sequence information as tuple (i.e. the "row" of the gs dataframe as namedtuple) and the
# data for each iteration.
# Note that the index of the data is not changed.
# Hence, we recommend using `iloc` to access the data (`iloc[0]` will return the first sample of the gait sequence).
#
# Using our example data and gs, we can iterate over the data as follows:
from mobgap.pipeline import iter_gs

for gs, data in iter_gs(long_trial.data_ss, long_trial_gs):
    # Note that the key to access the id is called "wb_id" here, as we loaded the WB from the reference system.
    # If this is an "actual" gait sequences, as calculated by one of the GSD algorithms, the key would be "gs_id".
    print("Gait Sequence: ", gs)
    print("Expected N-samples in gs: ", gs.end - gs.start)
    print("N-samples in gs: ", len(data))
    print("First sample of gs:\n", data.iloc[0], end="\n\n")

# %%
# .. note:: The ``gs`` named-tuples returned by the iterator is of type ``GaitSequence``.
#           It contains the fields ``id``, ``start``, and ``end`` in this order.
#           When using the named access the ``id`` field corresponds to either the ``gs_id`` or ``wb_id`` of the input
#           dataframe, depending on what type of list was provided.
#
# You can see that this way it is pretty easy to iterate over the data.
# However, if you are planning to run calculations on the data, you need to aggregate the results yourself.
# If you are planning to collect multiple pieces of results, this can become cumbersome.
# See the is `tpcp example <https://tpcp.readthedocs.io/en/latest/auto_examples/recipies/_04_typed_iterator.html>`__
# for more information about this.
# Therefore, we also provide an Iterator Class based on :class:`~tpcp.misc.TypedIterator`.
#
# Class based Interface
# ---------------------
# .. note:: Learn more about the general approach of using :class:`~tpcp.misc.TypedIterator` classes in this
#           `tpcp example <https://tpcp.readthedocs.io/en/latest/auto_examples/recipies/_04_typed_iterator.html>`__.
#
# Compared to the functional interface, the class interface attempts to also solve the problem of collecting the
# and aggregating results that you produce per GS.
# In a typical pipeline you might want to calculate the initial contacts, cadence, stride length, and gait speed for
# each gait sequence.
# With the class based interface, you can easily collect all of these results and then aggregate them into one
# predefined data structure.
#
# The class based interface can be used in two ways.
# First in the "default" configuration, which is set up to work with the typical calculations and results that you would
# expect from a typical processing pipeline.
# And second, in a custom way, where you need to define expected "results" per iteration yourself.
#
# The simple case
# ---------------
# The simple case basically no more setup as the functional interface.
# However, it assumes that your results are a subset of initial contacts, cadence, stride length, and gait speed, and
# that all of them are stored in the expected mobgap datatypes (aka pandas dataframes).
# The iterator will then automatically aggregate the results the dataframes per iteration into one combined dataframe,
# handling the sample offsets of the gait sequences for you.
#
# Below we will show how this works, by "simulating" the calculation of some initial contacts and cadence.
#
# We start by setting up an iterator object.
# We can leave everything at the default values, as we do not need any custom aggregation functions.
from mobgap.pipeline import GsIterator

iterator = GsIterator()
dt = iterator.data_type

# %%
# The default result datatype per iteration is defined as follows:
import inspect

from IPython.core.display_functions import display

display(inspect.getsource(dt))

# %%
# This means you are only allowed to use the available attributes.
# But, you don't need to specify all of them.
# Below we will only "calculate" the initial contacts and cadence.
#
# In each iteration the iterator will give us a tuple of the gait sequence information, the data for the iteration, and
# a new empty result object.
from mobgap.utils.conversions import as_samples

for (gs, data), result in iterator.iterate(long_trial.data_ss, long_trial_gs):
    # Now we can just "calculate" the initial contacts and set it on the result object.
    result.ic_list = pd.DataFrame(np.arange(0, len(data), 100, dtype="int64"), columns=["ic"]).rename_axis(
        index="step_id"
    )
    # For cadence, we just set a dummy value to the wb_id for each 1 second bout of the data.
    n_seconds = int(len(data) // long_trial.sampling_rate_hz)
    result.cad_per_sec = pd.DataFrame(
        [gs.id] * n_seconds,
        columns=["cad_spm"],
        index=as_samples(np.arange(0, n_seconds) + 0.5, long_trial.sampling_rate_hz),
    ).rename_axis(index="sec_center_samples")

# %%
# After the iteration, we can access the aggregated results using the `results_` property of the iterator
iterator.results_.ic_list

# %%
# We can see that we only get a single dataframe with all the results.
# And all ICs are offset, so that they are relative to the start of the recording and not the start of the gait
# sequence anymore.
#
# For the cadence value, the index represents the sample of the center of the second the cadence value belongs to.
# This value was originally relative to the start of the GS.
# We can see that in the aggregated results this is transformed back to be relative to the start of the recording.
iterator.results_.cad_per_sec


# %%
# But what to do, if you don't want to use the default result datatype?
#
# Custom Results
# --------------
#
# This requires a little bit more setup.
# First we need to decide what results we expect.
# This is done by defining a dataclass that represents the results.
#
# Here we create a new dataclass that only expect two dummy results, but you can add as many as you want.
# You could also subclass the default dataclass and just add the additional results.
#
# The first result here is ``n_samples`` which is just a dummy results indicating the number of samples the data has.
# The second result is ``filtered_data`` (we will just add some dummy data here).
# This is expected to be a pd.DataFrame to demonstrate that you can also return more complex results.
from dataclasses import dataclass


@dataclass
class ResultType:
    n_samples: int
    filtered_data: pd.DataFrame


# %%
# For each iteration (i.e. for each gait sequence), we will create one instance of this dataclass.
# The list of these instances will be available as the `raw_results_` attribute of the iterator.
#
# We can also decide to aggregate the results.
# We provide some default aggregations functions (see ``GsIterator.DEFAULT_AGGREGATORS``), that you could use.
# However, here we will create our own aggregation function.
#
# It might be nice to turn the ``n_samples`` into a pandas series with the gs identifier as index.
# For this we define an aggregation function that expects the list of ``TypedIteratorResultTuple``.
# These are named tuples of the following shape:
from mobgap.pipeline._overwrite_typed_iterator import TypedIteratorResultTuple

display(inspect.getsource(TypedIteratorResultTuple))


# %%
# The type of the ``input`` and the ``result`` depend on the dataclass you defined and the iterator you use.
# For the gait sequence iterator the input-type will be ``tuple[GaitSequence, pd.DataFrame]`` and the result-type will
# the dataclass you defined.
# The other arguments provide additional context, that might be needed in advanced cases (see lower down in this
# example).
#
# To simplify typing of functions that use these types, we provide ``GsIterator.IteratorResult`` which already has the
# input type bound and is generic with respect to the output type.
# We can see in the function below how to use it.
#
# As mentioned, an aggregation function will get a list of these named tuples.
# Note, that the values get passed the entire result object and that parts of the result objects might be ``NOT_SET``.
# To filter out the ``NOT_SET`` values and replace the ``result`` attribute with just one specific value, we provide
# the ``GsIterator.filter_iterator_results`` function (see below).
#
# With that, out aggregate function, takes the gs-id from the inputs and the n_samples from the results and creates a
# pandas series with the gs-id as index and the n_samples as values.
def aggregate_n_samples(values: list[GsIterator.IteratorResult[ResultType]]):
    non_null_results: GsIterator.IteratorResult[int] = GsIterator.filter_iterator_results(values, "n_samples")
    results = {r.input[0].id: r.result for r in non_null_results}
    return pd.Series(results, name="N-Samples")


aggregations = [("n_samples", aggregate_n_samples)]

# %%
# Now we can create an instance of the iterator.
# Note, that if we want to correctly infer the result type, we need to use the somewhat weird square bracket-typing
# syntax, when creating the iterator.
# This will allow to autocomplete the attributes of the result type.

custom_iterator = GsIterator[ResultType](ResultType, aggregations=aggregations)

# %%
# Iterating over the iterator now provides us the row from the gait sequence list (which we ignore here), the data for
# each iteration, and the empty result object, we can fill up each iteration.

for (_, data), custom_result in custom_iterator.iterate(long_trial.data_ss, long_trial_gs):
    # We just calculate the length, but you can image any other calculation here.
    # Then we just set the result.
    custom_result.n_samples = len(data)
    # For the "filtered" data we just subtract 1 form the input
    custom_result.filtered_data = data - 1

# %%
# Then we can easily inspect the aggregated results.
# Note, while the typing system can correctly infer the available attributes of the result object, the typing of the
# attributes might be wrong as Python can not infer the types based on the aggregations.
# We have to explicitly cast the value if we care about the type-correctness,
from typing import cast

n_samples = cast(int, custom_iterator.results_.n_samples)
n_samples

# %%
# For the filtered data, we did not apply any aggregation and hence just get a list of all results.
filtered_data = cast(list[pd.DataFrame], custom_iterator.results_.filtered_data)
filtered_data
