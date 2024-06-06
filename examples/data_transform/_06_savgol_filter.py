"""
Savitzky-Golay Filter Example
=============================
The Savitzky-Golay filter is a type of low-pass filter that can be used to smooth a signal while preserving its
original shape.
It fits a polynomial to a window of data and uses the polynomial to estimate the value of the central point of the
window.
The window is then moved along the data and the process is repeated.

We provide a class based implementation of the Savitzky-Golay filter in the
:class:`~mobgap.data_transform.SavgolFilter` class.
Compared to the version provided with scipy (:func:`scipy.signal.savgol_filter`), our implementation allows to specify
all parameters independently of the sampling rate of the data.
This should allow to apply the same filter to different datasets, while maintaining the same filter characteristics.
"""

import matplotlib.pyplot as plt
from mobgap.data import LabExampleDataset
from mobgap.data_transform import SavgolFilter

# %%
# Loading some example data
# -------------------------
example_data = LabExampleDataset()
ha_example_data = example_data.get_subset(cohort="HA")
single_test = ha_example_data.get_subset(
    participant_id="002", test="Test5", trial="Trial2"
)
data = single_test.data_ss

data.head()

# %%
# Applying the Savitzky-Golay filter
# ----------------------------------
# We need to specify the window length and the polynomial order.
# The window length is given in seconds and the polynomial order is given as a relative quantity to the effective
# window length in samples.
#
# For example, the values below at a sampling rate of 100 Hz would result in an effective window length of 20 samples
# and a polynomial order of 7.
savgol_filter = SavgolFilter(window_length_s=0.21, polyorder_rel=1 / 3)

savgol_filter.filter(data, sampling_rate_hz=single_test.sampling_rate_hz)
filtered_data = savgol_filter.filtered_data_

filtered_data.head()

# %%
# We can also access the effective window length and the polynomial order in samples that were calculated from the
# provided window length and the sampling rate.
savgol_filter.window_length_samples_, savgol_filter.polyorder_

# %%
# We will plot the filtered data together with the original data.
fig, axs = plt.subplots(3, 1, sharex=True)
for ax, col in zip(axs, ["gyr_x", "gyr_y", "gyr_z"]):
    ax.set_title(col)
    data[col].plot(ax=ax, label="Original")
    filtered_data[col].plot(ax=ax, label="CWT-filtered")

axs[0].set_xlim(data.index[300], data.index[500])
axs[0].legend()
fig.show()

plt.show()
