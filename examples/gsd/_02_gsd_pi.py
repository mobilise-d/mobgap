"""Example_GSD Implementation"""

from gaitlink.data import LabExampleDataset
from gaitlink.gsd import GsdParaschivIonescu

# Load data
data_all = LabExampleDataset()  # Data is in m/s2
long_trial = data_all.get_subset(cohort="MS", participant_id="001", test="Test11", trial="Trial1")


imu_data = long_trial.data["LowerBack"]


# Get GSD_LowBackAcc inputs
fs = long_trial.sampling_rate_hz

# Run GSD_LowBackAcc
gsd_output = GsdParaschivIonescu().detect(imu_data, sampling_rate_hz=fs).gs_list_

print(gsd_output)
