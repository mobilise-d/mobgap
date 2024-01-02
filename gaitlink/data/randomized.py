from gaitlink.data import LabExampleDataset
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

# Load the data
example_data = LabExampleDataset()
ha_example_data = example_data.get_subset(cohort="HA")
single_test = ha_example_data.get_subset(participant_id="002", test="Test11", trial="Trial1")
df = single_test.data["LowerBack"]

# Original and target sampling rates
original_sampling_rate = 100
target_sampling_rate = 40

# Calculate the downsampling factor
downsampling_factor = original_sampling_rate // target_sampling_rate

# Perform the downsampling
resampled_data = signal.resample(df, len(df) // downsampling_factor)
 
# Create a DataFrame from the resampled data
resampled_df = pd.DataFrame(data=resampled_data, columns=df.columns)

original_sampling_rate_ = 40
target_sampling_rate_ = 120

upsampling_factor = target_sampling_rate / original_sampling_rate
print(upsampling_factor)
resampled_data_ = signal.resample(resampled_data, len(resampled_data) // 3)

# Create a DataFrame from the resampled data
resampled_df_ = pd.DataFrame(data=resampled_data_, columns=df.columns)



# Plot the "gyr" data from the resampled DataFrame
resampled_df_.filter(like="acc").plot()
plt.title("Upsampled to 120 Hz acc data")
plt.show()
