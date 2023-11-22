import matplotlib.pyplot as plt
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
savgol_filter = SavgolFilter(window_length=5, polyorder=2)

# Perform the Savitzky-Golay filtering operation by calling the transform method
smoothed_data = savgol_filter.transform(df)

# Access the smoothed data
smoothed_acc_data = smoothed_data.transformed_data_  # Assuming transformed_data_ is a DataFrame

# Plot the raw 'acc' data
df.filter(like="acc").plot(label="Raw Data")
smoothed_acc_data_df = pd.DataFrame(smoothed_acc_data, columns=df.columns)

# Filter the smoothed data to only include the 'acc' and 'gyro' columns (replace with your column names)
smoothed_acc_gyro_data = smoothed_acc_data_df.filter(like="acc")
# Plot the smoothed 'acc' data
smoothed_acc_data_df.filter(like="acc").plot(label="Smoothed Data")
plt.title("Raw and Smoothed Data")
plt.legend()
plt.show()
