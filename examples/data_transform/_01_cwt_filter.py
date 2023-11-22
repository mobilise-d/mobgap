from scipy.signal import morlet
from gaitlink.data_transform._cwt_filter import CwtFilter
import numpy as np
import matplotlib.pyplot as plt
# Generate some example data (replace this with your actual data)
x = np.linspace(0, 1, 1000)
y = np.sin(2 * np.pi * 5 * x)  # Example sine wave

# Create an instance of the CwtFilter with the morlet wavelet
cwt_filter = CwtFilter(wavelet=morlet, width=np.arange(1, 31))

# Perform the CWT operation by calling the transform method
cwt_result = cwt_filter.transform(y)

# Access the transformed data
transformed_data = cwt_result.transformed_data_

# Plot the original and transformed data


plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(x, y, label="Original Data")
plt.title("Original Data")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(2, 1, 2)
plt.imshow(np.abs(transformed_data), aspect='auto', extent=[0, 1, 1, 31], cmap='jet', origin='lower')
plt.title("Continuous Wavelet Transform")
plt.xlabel("Time")
plt.ylabel("Scale")
plt.colorbar(label="Magnitude")

plt.tight_layout()
plt.show()
