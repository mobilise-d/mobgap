"""
Gait Sequence Iterator
======================

As part of most pipelines, we need to iterate over the gait sequences to apply all further algorithms to them
individually.
This can be a bit cumbersome, as we need to iterate over the data and aggregate the results at the same time.
Hence, we provide some helpers for that.

We provide two ways of iterating.
The first one, only handles the iteration and does not aggregate the results.
The second approach attempts to also support you in aggregating the results.
"""

# %%
# Getting Some Example Data
# -------------------------
# TODO: Simplify that once we have better loading methods
import pandas as pd

from gaitlink.data import LabExampleDataset


def load_reference_gs(datapoint):
    return (
        (
            pd.DataFrame.from_records(
                [
                    {"start": wb["Start"], "end": wb["End"], "gs_id": f"gs_{i}"}
                    for i, wb in enumerate(datapoint.reference_parameters_["wb"])
                ]
            ).set_index("gs_id")
            * datapoint.sampling_rate_hz
        )
        .round()
        .astype(int)
    )


lab_example_data = LabExampleDataset(reference_system="INDIP")
long_trial = lab_example_data.get_subset(cohort="MS", participant_id="001", test="Test11", trial="Trial1")
long_trial_gs = load_reference_gs(long_trial)

long_trial_gs

# %%
# Simple Functional Interface
# ---------------------------
# We provide the :func:`~gaitlink.pipeline.iter_gs` function to iterate over the gait sequences.
# It simply takes the data and the gait sequence list and cuts the data accordingly to iterate over it.
# Using our example data and gs, we can iterate over the data as follows:
from gaitlink.pipeline import iter_gs

for gs_id, data in iter_gs(long_trial.data["LowerBack"], long_trial_gs):
    print("Gait Sequence: ", gs_id)
    print("Expected N-samples in gs: ", long_trial_gs.loc[gs_id].end - long_trial_gs.loc[gs_id].start)
    print("N-samples in gs: ", len(data))

# %%
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
# The class based interface requires a little bit more setup, but allows to easily aggregate results.
# First we need to decide what results we expect.
# This is done by defining a dataclass that represents the results.
#
# Here we only expect two dummy results, but you can add as many as you want.
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
# In this case, it might be nice to turn the ``n_samples`` into a pandas series with the gs identifier as index.
# For this we define an aggregation function that expects the list of inputs and the list of results as inputs.


def aggregate_n_samples(inputs, results):
    gs_ids, _ = zip(*inputs)
    return pd.Series(results, index=gs_ids, name="N-Samples")


aggregations = [("n_samples", aggregate_n_samples)]

# %%
# Now we can create an instance of the iterator.
from gaitlink.pipeline import GsIterator

iterator = GsIterator(ResultType, aggregations=aggregations)

# %%
# Iterating over the iterator now provides us the row from the gait sequence list (which we ignore here), the data for
# each iteration, and the empty result object, we can fill up each iteration.

for (_, data), result in iterator.iterate(long_trial.data["LowerBack"], long_trial_gs):
    # We just calculate the length, but you can image any other calculation here.
    # Then we just set the result.
    result.n_samples = len(data)
    # For the "filtered" data we just substract 1 form the input
    result.filtered_data = data - 1

# %%
# Then we can easily inspect the aggregated results.
iterator.n_samples_

# %%
# For the filtered data, we did not apply any aggregation and hence just get a list of all results.
iterator.filtered_data_
