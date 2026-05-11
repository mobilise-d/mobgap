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
from mobgap.data import LabExampleDataset
from mobgap.utils.conversions import to_body_frame
from re_orientation.re_orientation import ReorientationMethodDM

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
# Like most algorithms, the algorithm requires the data to be in body frame coordinates.
# As we know the sensor was well aligned, we can just use ``to_body_frame`` to transform the data.

single_test = example_data.get_subset(
    cohort="HA", participant_id="001", test="Test11", trial="Trial1"
)

reference_wbs = single_test.reference_parameters_.wb_list

# Including only 1 WB as the example
start = reference_wbs.iloc[2]["start"]
end = reference_wbs.iloc[2]["end"]

imu_data = to_body_frame(single_test.data.get("LowerBack"))
imu_data = imu_data.reset_index(drop=True)

first_WB = imu_data.loc[start:end]

# %%
# Introducing artificial misorientation
# --------------------------------------
# To demonstrate the algorithm, we artificially introduce a misorientation.
# Below we flip IS and mediolateral axes (family 2).

# Flipping IS and mediolateral (family 2)
first_WB["acc_is"] = -first_WB["acc_is"]
first_WB["gyr_is"] = -first_WB["gyr_is"]
first_WB["acc_ml"] = -first_WB["acc_ml"]
first_WB["gyr_ml"] = -first_WB["gyr_ml"]

print(first_WB)

# %%
# Visualising the misoriented walking bout
# -----------------------------------------
# We can visualise the walking bout before correction.

fig, ax = plt.subplots()

ax.plot(first_WB["acc_is"].to_numpy(), label="acc_is")
ax.plot(first_WB["acc_ml"].to_numpy(), label="acc_ml")
ax.plot(first_WB["acc_pa"].to_numpy(), label="acc_pa")

ax.legend()
fig.show()

# %%
# Applying the reorientation algorithm
# -------------------------------------
# Below we apply the ReorientationMethodDM algorithm to the misoriented walking bout.
# We use the 'full' method which applies all three stages.

# Calling ReorientationMethodDM
Reoriented = ReorientationMethodDM(method='full')
result = Reoriented.detect_correct(first_WB)
print(result)
corrected = result.data_corrected

print(corrected)

# %%
# Visualising the corrected walking bout
# ---------------------------------------
# After correction, we can visualise the corrected data.

fig, ax = plt.subplots()

ax.plot(corrected["acc_is"].to_numpy(), label="acc_is")
ax.plot(corrected["acc_ml"].to_numpy(), label="acc_ml")
ax.plot(corrected["acc_pa"].to_numpy(), label="acc_pa")

ax.legend()
fig.show()