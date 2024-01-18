import numpy as np
from scipy.ndimage import gaussian_filter
from gaitlink.data_transform import GaussianFilter  # Update the import path accordingly

# Example usage
# Assume you have some multidimensional data stored in a NumPy array 'your_data'
your_data = np.random.rand(10, 10, 10)  # Replace this with your actual data

# Create an instance of the GaussianFilter class with a specified sigma value
sigma_value = 1.0  # Replace this with your desired sigma value
gaussian_filter_instance = GaussianFilter(sigma=sigma_value)

# Perform the blurring operation by calling the transform method
blurred_data = gaussian_filter_instance.transform(your_data)

# Access the blurred data
blurred_result = blurred_data.transformed_data_

# Compare the original and blurred data (you can use visualization tools suitable for your data)
# For example, you can use matplotlib for 2D data
import matplotlib.pyplot as plt

# Plot the original and blurred data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(your_data[:, :, 0], cmap='viridis')
plt.title('Original Data')

plt.subplot(1, 2, 2)
plt.imshow(blurred_result[:, :, 0], cmap='viridis')
plt.title('Blurred Data')

plt.show()
