from mobgap.data import LabExampleDataset
from gaitmap.trajectory_reconstruction import MadgwickAHRS
import numpy as np
import pandas as pd

# The below tests the Madgwick algorithm from gaitmap on the lab data

example_data = LabExampleDataset(reference_system="INDIP", reference_para_level="wb")

single_test = example_data.get_subset(cohort="HA", participant_id="001", test="Test11", trial="Trial1")
imu_data = single_test.data_ss
IMU = imu_data.iloc[0:50, :]

#testing Madgwick from gaitmap
quaternion_list = []
for index, row in IMU.iterrows():
    row = pd.DataFrame([row.values], columns=row.index)
    sampling_rate_hz = 100
    # the below initial orientation is the same as the one in the MATLAB with the real part placed at the end
    # MATLAB:[0.8 0.75 0 0]/norm([0.8 0.75 0 0])
    mad = MadgwickAHRS(beta=0.1, initial_orientation=np.array([0.68394113, 0, 0, 0.7295372]))
    mad = mad.estimate(row, sampling_rate_hz=sampling_rate_hz)
    orientations = mad.orientation_
    #gaitmap uses xyzw format
    quaternion = orientations.iloc[:, [3, 0, 1, 2]]
    quaternion_list.append(quaternion.iloc[1])

quaternions = pd.DataFrame(quaternion_list)
print(quaternions)

quaternions.to_csv('gaitmap.csv', index=False)

