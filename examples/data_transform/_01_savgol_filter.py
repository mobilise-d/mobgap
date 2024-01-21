"""
Savitzky-Golay Filter Example
===============
"""

from gaitlink.data import LabExampleDataset
from gaitlink.data_transform._savgol_filter import SavgolFilter
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
example_data = LabExampleDataset()
ha_example_data = example_data.get_subset(cohort="HA")
single_test = ha_example_data.get_subset(participant_id="002", test="Test11", trial="Trial1")
df = single_test.data["LowerBack"]

# Create an instance of the SavgolFilter class with your desired parameters
savgol_filter = SavgolFilter(window_length=10, polyorder=2)

# Perform the Savgol filtering operation by calling the transform method
smoothed_data = savgol_filter.transform(df)

# Access the smoothed data
smoothed_acc_data = smoothed_data.transformed_data_  # Assuming transformed_data_ is a NumPy array

# Convert NumPy array to pandas DataFrame
smoothed_acc_data_df = pd.DataFrame(smoothed_acc_data, columns=df.columns)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Plot the raw 'acc' data
df.filter(like="acc").plot(ax=axs[0], label="Raw Data")
axs[0].set_title("Raw Data")
axs[0].legend()

# Plot the smoothed 'acc' data
smoothed_acc_data_df.filter(like="acc").plot(ax=axs[1], label="Smoothed Data")
axs[1].set_title("Smoothened Data")
axs[1].legend()

plt.show()
