"""
General Filter Introduction
===========================

One of the most common data transformations is filtering.
In mobgap various types of filters are used to clean data and enhance signal features.
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

Below we show two simple examples.

First, we show how to apply the :class:`~mobgap.data_transform.EpflDedriftedGaitFilter` to some data and then
we demonstrate a simple butterworth filter.

We load some example data to apply the filters to.

"""

from mobgap.data import LabExampleDataset
from mobgap.data_transform import Resample

data_point = LabExampleDataset().get_subset(
    cohort="HA", participant_id="002", test="Test5", trial="Trial2"
)
example_data = data_point.data_ss
sampling_rate_hz = data_point.sampling_rate_hz

example_data

# %%
# EpflDedriftedGaitFilter
# -----------------------
# The EpflDedriftedGaitFilter is a filter that is specifically designed to remove drift from gait data.
# However, it only works on data that is sampled at 40 Hz.
# Hence, we need to resample the data first.

example_data_resampled = (
    Resample(target_sampling_rate_hz=40)
    .transform(example_data, sampling_rate_hz=sampling_rate_hz)
    .transformed_data_
)
example_data_resampled

# %%
# We can now apply the filter to the data.
from mobgap.data_transform import EpflDedriftedGaitFilter

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

# %%
# Butterworth filter
# ------------------
# The Butterworth filter is a generic filter that can be used to filter data.
# However, its interface is identical to all other filter methods, making it easy to exchange filters.
#
# Most of the parameters of the filter are forwarded to the :func:`scipy.signal.butter` function.
# See the documentation of that function for more information.
#
# Our implementation uses either :func:`scipy.signal.filtfilt` or :func:`scipy.signal.lfilter` depending on the
# ``zero_phase`` parameter.
# Here, we will use the default (``zero_phase=True``), which means that the filter is applied twice, once forward and
# once backward.
from mobgap.data_transform import ButterworthFilter

butterworth_filter = ButterworthFilter(
    order=4, cutoff_freq_hz=2.0, zero_phase=True
)

butterworth_filter.filter(example_data_resampled, sampling_rate_hz=40)

filtered_data = butterworth_filter.filtered_data_
filtered_data

# %%
# We can again plot the filtered data to see the effect of the filter.
example_data_resampled["gyr_y"].plot(label="Original data")
filtered_data["gyr_y"].plot(label="Filtered data")
plt.legend()

plt.show()
