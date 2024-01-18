from scipy.signal import ricker
import numpy as np
from gaitlink.data import LabExampleDataset
import matplotlib.pyplot as plt
from gaitlink.data_transform import CwtFilter

# Load example data
example_data = LabExampleDataset()
ha_example_data = example_data.get_subset(cohort="HA")
single_test = ha_example_data.get_subset(participant_id="002", test="Test11", trial="Trial1")
df = single_test.data["LowerBack"]

# Define your wavelet function and width
wavelet = ricker
width = 10.0

# Create an instance of CwtFilter
cwt_filter = CwtFilter(wavelet=wavelet, width=width)

# Transform the data using CwtFilter
transformed_data = cwt_filter.transform(df, widths=[width])


# Access the transformed data
# transformed_data = cwt_filter.transformed_data_


print(transformed_data.data)


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(df.index, df.values, label='Original Data')
plt.title('Original Data')
plt.xlabel('Time')
plt.ylabel('Signal Value')
plt.legend()

# Plot the transformed data
plt.subplot(1, 2, 2)
num_scales = transformed_data.data.shape[0]  # Assuming the first dimension is scales
time_points = df.index.to_numpy()

plt.imshow(np.abs(transformed_data.data), aspect='auto', extent=[time_points[0], time_points[-1], 0, num_scales],
           cmap='PRGn', origin='lower')
plt.title('CWT Transformed Data')
plt.xlabel('Time')
plt.ylabel('Scale')
plt.colorbar(label='Magnitude')

plt.tight_layout()
plt.show()