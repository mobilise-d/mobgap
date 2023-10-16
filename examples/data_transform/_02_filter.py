"""
Filtering data
==============

One of the most common data transformations is filtering.
In gaitlink various types of filters are used to clean data and enhance signal features.
Important filters are implemented using the :class:`~tpcp.base.data_transform.BaseFilter` class, which supports
a consistent interface for filtering data.
See the :ref:`generic data transforms example <generic_data_transforms>` for more information.

Working with filters
--------------------
Filters can be used like any other data transform.
This means they support the `transform` method and results are stored in the `transformed_data_` attribute.
However, to make using filters even more intuitive, they also support the `filter` method and the `filtered_data_`
attribute, which are simply aliases for `transform` and `transformed_data_` respectively.
All filters use :class:`~tpcp.data_transform.base.BaseFilter` as a base class, which implements these aliases.

Below we show a simple example of how to apply the :class:`~gaitlink.data_transform.EpflDedriftedGaitFilter` to some data.
This filter is used as part of the pre-processing of many of the gaitlink algorithms.

We load some example data to apply the filter to.
Note that the filter is designed to only work on 40 Hz data.
Hence, we need to resample the data first.
"""
import pandas as pd
from scipy.signal import resample

from gaitlink.data import LabExampleDataset

data_point = LabExampleDataset().get_subset(cohort="HA", participant_id="002", test="Test5", trial="Trial2")
example_data = data_point.data["LowerBack"]
sampling_rate_hz = data_point.sampling_rate_hz

# TODO: Update once a resample method is available
example_data_resampled = pd.DataFrame(
    resample(example_data, num=int(len(example_data) * 40 / sampling_rate_hz), axis=0), columns=example_data.columns
)
example_data_resampled

# %%
# We can now apply the filter to the data.
from gaitlink.data_transform import EpflDedriftedGaitFilter

epfl_filter = EpflDedriftedGaitFilter()
epfl_filter.filter(example_data_resampled, sampling_rate_hz=40)

# %%
# The filtered data is now stored in the `filtered_data_` attribute.
epfl_filter.filtered_data_

# %%
# We can plot the filtered data to see the effect of the filter.
import matplotlib.pyplot as plt

example_data_resampled["gyr_y"].plot(label="Original data")
epfl_filter.filtered_data_["gyr_y"].plot(label="Filtered data")
plt.legend()

plt.show()
