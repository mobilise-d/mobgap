"""
Gaussian Filter data
===============
"""
import matplotlib.pyplot as plt

from gaitlink.data import LabExampleDataset
from gaitlink.data_transform import GaussianFilter  # Update the import path accordingly

# Example usage
example_data = LabExampleDataset()
ha_example_data = example_data.get_subset(cohort="HA")
single_test = ha_example_data.get_subset(participant_id="002", test="Test11", trial="Trial1")
df = single_test.data["LowerBack"]

# Assume you have some multidimensional data stored in a NumPy array 'your_data'
# your_data = np.random.rand(10, 10, 10)  # Replace this with your actual data

# Create an instance of the GaussianFilter class with a specified sigma value
sigma_value = 1.0  # Replace this with your desired sigma value
gaussian_filter_instance = GaussianFilter(sigma=sigma_value)

# Perform the blurring operation by calling the transform method
blurred_data = gaussian_filter_instance.filter(df)

# Access the blurred data
blurred_result = blurred_data.filtered_data_

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
