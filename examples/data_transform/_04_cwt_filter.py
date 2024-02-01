"""
Continuous Wavelet Transform (CWT) Example
===============

"""

import matplotlib.pyplot as plt
from scipy.signal import ricker

from gaitlink.data import LabExampleDataset
from gaitlink.data_transform import CwtFilter

# Load example data
example_data = LabExampleDataset()
ha_example_data = example_data.get_subset(cohort="HA")
single_test = ha_example_data.get_subset(participant_id="002", test="Test11", trial="Trial1")
df = single_test.data["LowerBack"]

# Define your wavelet function and width
wavelet = ricker
cwt_filter = CwtFilter(wavelet=wavelet)

# Transform the data using CwtFilter
cwt_filter.filter(df)

# Access the transformed data
transformed_data = cwt_filter.filtered_data_
print(transformed_data)


# Plot original and fitlered data together
fig, ax = plt.subplots()
df.reset_index(drop=True).plot(ax=ax)
transformed_data.add_suffix("_filtered").reset_index(drop=True).plot(ax=ax)
ax.set_xlim(0, 1000)
plt.show()
