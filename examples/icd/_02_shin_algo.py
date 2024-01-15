"""
Shin Algo
=========
"""

from matplotlib import pyplot as plt

from gaitlink.data import LabExampleDataset
from gaitlink.icd import IcdShinImproved

# %%
# Loading data
# ------------
example_data = LabExampleDataset(reference_system="INDIP", reference_para_level="wb")


ha_example_data = example_data.get_subset(cohort="HA")
single_test = ha_example_data.get_subset(participant_id="001", test="Test5", trial="Trial2")
imu_data = single_test.data["LowerBack"]

sampling_rate_hz = single_test.sampling_rate_hz
ref_ics = single_test.reference_parameters_.initial_contacts


# %%
# Applying the algorithm
# ----------------------
icd = IcdShinImproved(axis="x")

icd.detect(imu_data, sampling_rate_hz=sampling_rate_hz)

icd.ic_list_

# %%
# Plotting the results
# --------------------
plt.plot(icd.data["acc_x"].reset_index(drop=True))
plt.plot(icd.ic_list_, icd.data["acc_x"].iloc[icd.ic_list_["ic"]], "x")
plt.plot(ref_ics["ic"], icd.data["acc_x"].iloc[ref_ics["ic"]], "o")




