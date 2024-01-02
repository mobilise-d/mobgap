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

# Calculate the upsampling and downsampling factors
upsampling_factor = target_sampling_rate
downsampling_factor = original_sampling_rate

# Perform the resampling using signal.resample_poly
resampled_data = signal.resample_poly(df, up=upsampling_factor, down=downsampling_factor, axis=0)

# Create a DataFrame from the resampled data
resampled_df = pd.DataFrame(data=resampled_data, columns=df.columns)

# Plot the "gyr" data from the resampled DataFrame
resampled_df.filter(like="gyr").plot()
plt.title("Resampled poly gyr data")
plt.show()
