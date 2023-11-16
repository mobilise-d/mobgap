""" Example_GSD Implementation"""

from gaitlink.data import LabExampleDataset
from gaitlink.gsd_b.gsd_b import GSD_LowBackAcc

# Load data
data_all = LabExampleDataset()  # Data is in m/s2
print(data_all.grouped_index.index.to_list())

# Iterate over the trials in data_all
for group in data_all.grouped_index:

    single_test = data_all.get_subset(group_labels=[group])  # Get a single trial

    imu_data = single_test.data["LowerBack"]
    acc = imu_data[[col for col in imu_data if col.startswith('acc')]]  # Select accelerometer columns

    # Get GSD_LowBackAcc inputs
    fs = single_test.sampling_rate_hz
    plot_results = True

    # Run GSD_LowBackAcc
    gsd_output = GSD_LowBackAcc(acc, fs, plot_results)

print('Done')
