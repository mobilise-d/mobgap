"""
Gaussian Smoothing
==================
A common low-pass filtering technique is Gaussian smoothing, which applies a Gaussian window to the data with a moving
average.
We provide a class based implementation of Gaussian smoothing in the :class:`GaussianFilter` class.
"""
import matplotlib.pyplot as plt

from gaitlink.data import LabExampleDataset
from gaitlink.data_transform import GaussianFilter  # Update the import path accordingly

# %%
# Loading some example data
# -------------------------
example_data = LabExampleDataset()
ha_example_data = example_data.get_subset(cohort="HA")
single_test = ha_example_data.get_subset(participant_id="002", test="Test5", trial="Trial2")
data = single_test.data["LowerBack"]

data.head()

# %%
# Applying the Gaussian filter
# ----------------------------
# We need to specify the standard deviation of the Gaussian window.
# Note, that we need to specify the standard deviation in seconds to make the parameters of the filter independent of
# the sampling rate of the data.
# It will be converted to samples internally.
gaussian_filter = GaussianFilter(sigma_s=0.1)
gaussian_filter.filter(data)
smoothed_data = gaussian_filter.filtered_data_

smoothed_data.head()

# %%


# Compare the original and blurred data (you can use visualization tools suitable for your data)
# For example, you can use matplotlib for 2D data


# Plot the original and blurred data
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(df.values, cmap="viridis")
# plt.title("Original Data")


plt.figure(figsize=(12, 4))

# Plotting the original data
plt.subplot(1, 2, 1)
plt.plot(df.index, df.values, label="Original Data")
plt.title("Original Data")
plt.xlabel("Time")
plt.ylabel("Signal Value")
plt.legend()

# Plotting the blurred data
plt.subplot(1, 2, 2)
plt.plot(blurred_result.index, blurred_result.values, label="Blurred Data")
plt.title("Blurred Data")
plt.xlabel("Time")
plt.ylabel("Signal Value")
plt.legend()
plt.show()
