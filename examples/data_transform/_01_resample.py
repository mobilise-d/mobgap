#Here's a quick implementation of the resample class

from gaitlink.data import LabExampleDataset
from gaitlink.data_transform._resample import Resample

# Now you can use the Resample class in this file

import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
# Load the data
example_data = LabExampleDataset()
ha_example_data = example_data.get_subset(cohort="HA")
single_test = ha_example_data.get_subset(participant_id="002", test="Test11", trial="Trial1")
df = single_test.data["LowerBack"]

print(df.head())

#Here's a quick implementation of the function

# Example usage
data_to_resample = df  # Replace with your actual data
current_sampling_rate = 100  # Replace with your actual sampling rate
target_sampling_rate = 60  # Replace with your desired target sampling rate

# Create an instance of the Resample class with the target sampling rate
resampler = Resample(target_sampling_rate)

# Perform the resampling operation by calling the transform method
resampled = resampler.transform(df, sampling_rate_hz=100)

resampled_gyr = resampled.transformed_data_.filter(like="gyr")
# Access the resampled data
# resampled_acc_data = resampled.transformed_data_["gyr"]
# # Filter the resampled data to only include the 'acc' and 'gyro' columns
# resampled_acc_gyro_data = resampled.transformed_data_.filter(like="gyr")
#
# # Plot the resampled 'acc' data
resampled_gyr.plot()
if target_sampling_rate > current_sampling_rate:
    title = "Upsampled to " + str(target_sampling_rate) + " Hz"

    plt.title(title)
    plt.show()
elif target_sampling_rate < current_sampling_rate:
    title = "Downsampled to " + str(target_sampling_rate) + " Hz"
    plt.title(title)
    plt.show()
else:
    plt.title("No change in data")
    plt.show()



