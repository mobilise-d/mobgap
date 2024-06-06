"""
Continuous Wavelet Transform (CWT) - Filter
===========================================

Continuous wavelet transform (CWT) is a time-frequency analysis method that can provide frequency information localized
in both time and frequency.
For this, wavelets of different scales, where each scale corresponds to a different frequency, are used to analyze the
signal.
When performing the CWT with just a single scale, the CWT becomes equivalent to a narrow bandpass filter, enhancing
the frequency content of the signal around the scale of the wavelet.

This approach is often used in time series analysis to enhance specific frequencies of interest.

In this example we will show how to apply a CWT as such a filter to a time series.
"""

import matplotlib.pyplot as plt
from mobgap.data import LabExampleDataset
from mobgap.data_transform import CwtFilter

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
# Initializing the CWT filter
# ---------------------------
# This requires us to define the mother wavelet and the center frequency of the bandpass filter.
# We specify the frequency rather than the scale, as the effect of the scale will be dependent on the sampling rate of
# the data.
#
# To replicate a filter (i.e. old Matlab code) that uses the scale, you can use the following formula to convert the
# scale to frequency:
# ``f = pywt.scale2frequency(wavelet, scale)/sampling_period``
#
# We use the ``pywt`` package to provide the wavelets.
# You can check `this page <https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html>`_ for available wavelets.
cwt_filter = CwtFilter(wavelet="gaus2", center_frequency_hz=10)

# %%
# Applying the CWT filter
# -----------------------
# We can apply the filter by calling the filter method and then access the filtered data via the ``filtered_data_``
# attribute.

cwt_filter.filter(data, sampling_rate_hz=single_test.sampling_rate_hz)
filtered_data = cwt_filter.filtered_data_
filtered_data.head()

# %%
# We can also get the actual scale that was calculated from the center frequency.
cwt_filter.scale_

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
