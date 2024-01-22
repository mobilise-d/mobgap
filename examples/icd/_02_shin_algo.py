"""
Shin Algo
=========

This example shows how to use the improved Shin algorithm and some examples on how the results compare to the original
matlab implementation.

"""
import numpy as np
from matplotlib import pyplot as plt
from gaitlink.data import LabExampleDataset
from gaitlink.ICD._shin_algo_improved import IcdShinImproved
import pandas as pd

# %%
# Loading data
# ------------
# .. note :: More infos about data loading can be found in the :ref:`data loading example <data_loading_example>`.
# We load example data from the lab dataset together with the INDIP reference system.
# We will use the INDIP output for initial contacts ("ic") as ground truth.

example_data = LabExampleDataset(reference_system="INDIP", reference_para_level="wb")

ha_example_data = example_data.get_subset(cohort="MS")
single_test = ha_example_data.get_subset(participant_id="001", test="Test5", trial="Trial1")
imu_data = single_test.data["LowerBack"]

#importing the start and end of each walking bout identified from the INDIP system (ground truth)
GS = single_test.reference_parameters_.walking_bouts
GS = [{"Start": row["start"], "End": row["end"]} for _, row in GS.iterrows()]
GS = pd.DataFrame.from_records(GS)

sampling_rate_hz = single_test.sampling_rate_hz
ref_ics = single_test.reference_parameters_.initial_contacts

# %%
# Applying the algorithm
# ----------------------
# Below we apply the shin algorithm to a lab trial.
# In this example, we use only the x-axis of the accelerometer data in order to be consistent with the INDIP IC detection.
# This will allow for comparisons in the next step.
# However, the algorithm can be applied to all axes, to be consistent with the matlab shin algorithm and cad2sec.

SD_Output = {'Start': [], 'End': [], 'IC': []}
BN = len(GS['Start'])
startvec = np.zeros(BN, dtype=int)
stopvec = np.zeros(BN, dtype=int)

for i in range(BN):
    startvec[i] = int(np.floor(GS['Start'][i]))
    stopvec[i] = int(np.floor(GS['End'][i]))
    icd = IcdShinImproved(axis="x")
    acc_data = imu_data.iloc[startvec[i]:stopvec[i], :]
    icd.detect(acc_data, sampling_rate_hz=sampling_rate_hz)
    SD_Output['IC'].append(icd.ic_list_['ic'])
    SD_Output['Start'].append(startvec[i])
    SD_Output['End'].append(stopvec[i])

# %%
# Plotting the results
# --------------------
# Plotting the output, reveals that python is less sensitive in detecting ICs.
# In addition, python might detect false positives in a few instances.

#Old version good for single bout only
# plt.plot(icd.data["acc_x"].reset_index(drop=True))
# plt.plot(icd.ic_list_, icd.data["acc_x"].iloc[icd.ic_list_["ic"]], "x")
# plt.plot(ref_ics["ic"], icd.data["acc_x"].iloc[ref_ics["ic"]], "o")
# plt.show()

#Updated the plot to show the ICs for each walking bout in files with multiple bouts
plt.plot(imu_data['acc_x'].reset_index(drop=True))
plt.plot(ref_ics["ic"], imu_data['acc_x'].iloc[ref_ics["ic"]], "o")
for start_value, ic_series in zip(SD_Output['Start'], SD_Output['IC']):
    IC = ic_series.tolist()
    Start = start_value
    IC_sample = np.array(IC) + Start
    plt.plot(IC_sample, imu_data['acc_x'].iloc[IC_sample], "x")
plt.show()