r"""
.. _reorientation_method_dm:

Reorientation Method DM
=======================

This example shows how to use the ReorientationMethodDM algorithm to detect and correct persistent IMU
misorientation in lower-back-worn devices.

"""

# %%
# Import useful modules and packages
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from mobgap._gaitmap.utils.rotations import flip_dataset
from mobgap.data import LabExampleDataset
from mobgap.re_orientation import ReorientationMethodDM

# %%
# Loading some example data
# -------------------------
# .. note :: More information about data loading can be found in the :ref:`data loading example <data_loading_example>`.
#
# We load example data from the lab dataset together with the INDIP reference system.

example_data = LabExampleDataset(
    reference_system="INDIP", reference_para_level="wb"
)

# %%
# Performance on a single lab trial
# ---------------------------------
# Below we apply the algorithm to a lab trial, where we extract a single walking bout.
#
# The reorientation algorithm expects sensor-frame input and returns body-frame output.

single_test = example_data.get_subset(
    cohort="HA", participant_id="001", test="Test11", trial="Trial1"
)

reference_wbs = single_test.reference_parameters_.wb_list
sampling_rate_hz = single_test.sampling_rate_hz

# Including only 1 WB as the example
start = reference_wbs.iloc[2]["start"]
end = reference_wbs.iloc[2]["end"]

imu_data = single_test.data.get("LowerBack")
imu_data = imu_data.reset_index(drop=True)

first_WB = imu_data.loc[start:end].copy()

# %%
# Introducing artificial misorientation
# -------------------------------------
# To demonstrate the algorithm, we artificially introduce a misorientation.
# Below we rotate the sensor frame by 180 degrees around the sensor z-axis (family 2).

first_WB = flip_dataset(first_WB, Rotation.from_euler("z", 180, degrees=True))

print(first_WB)

# %%
# Visualising the misoriented walking bout
# ----------------------------------------
# We can visualise the walking bout before correction.

fig, ax = plt.subplots()

ax.plot(first_WB["acc_x"].to_numpy(), label="acc_x")
ax.plot(first_WB["acc_y"].to_numpy(), label="acc_y")
ax.plot(first_WB["acc_z"].to_numpy(), label="acc_z")

ax.legend()
fig.show()

# %%
# Applying the reorientation algorithm
# ------------------------------------
# Below we apply the ReorientationMethodDM algorithm to the misoriented walking bout.
# We use the 'full' correction mode which applies all three stages.

reoriented = ReorientationMethodDM(correction_mode="full").detect_correct(
    first_WB, sampling_rate_hz=sampling_rate_hz
)

print(f"\nDetected orientation family: {reoriented.result_.family}")
print(f"Correction applied: {reoriented.result_.correction_applied}")
print(f"Correction action: {reoriented.result_.correction_action}")

corrected = reoriented.corrected_data_

# %%
# Visualising the corrected walking bout
# --------------------------------------
# After correction, we can access the corrected data via the ``corrected_data_`` attribute.

fig, ax = plt.subplots()

ax.plot(corrected["acc_is"].to_numpy(), label="acc_is")
ax.plot(corrected["acc_ml"].to_numpy(), label="acc_ml")
ax.plot(corrected["acc_pa"].to_numpy(), label="acc_pa")

ax.legend()
fig.show()
