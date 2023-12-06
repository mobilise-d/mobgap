""" Example_GSD Implementation"""
import pandas as pd

from gaitlink.data import LabExampleDataset
from gaitlink.gsd import GsdLowBackAcc

# Load data
data_all = LabExampleDataset()  # Data is in m/s2
long_trial = data_all.get_subset(cohort="HA", participant_id="001", test="Test11", trial="Trial1")


imu_data = long_trial.data["LowerBack"]


acc = imu_data[[col for col in imu_data if col.startswith("acc")]]  # Select accelerometer columns

# Get GSD_LowBackAcc inputs
fs = long_trial.sampling_rate_hz
plot_results = True

# Run GSD_LowBackAcc
gsd_output = GsdLowBackAcc(acc, fs, plot_results)

gsd_output = pd.DataFrame(gsd_output).rename(columns={"Start": "start", "End": "end"}).drop(columns="fs").astype(int)
print(gsd_output)
