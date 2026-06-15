r"""
.. _reorientation_method_dm:

Reorientation Method DM
=======================

This example shows how to use the ReorientationMethodDM algorithm to detect
and correct persistent IMU misorientation in lower-back-worn devices.

``ReorientationMethodDM`` corrects the most common mounting errors for
lower-back-worn, flat rectangular sensors. It assumes that one of the large
flat sensor surfaces is mounted against the body and that, under correct
mounting, sensor x points along IS, sensor y points along ML, and sensor z
points along PA. The method corrects 90 deg and 180 deg mounting rotations,
not arbitrary small misalignments.

"""

# %%
# Import useful modules and packages
import matplotlib.pyplot as plt
from mobgap._gaitmap.utils.rotations import flip_dataset
from mobgap.data import LabExampleDataset
from mobgap.re_orientation import ReorientationMethodDM
from scipy.spatial.transform import Rotation

# %%
# Loading some example data
# -------------------------
# .. note ::
#    More information about data loading can be found in the
#    :ref:`data loading example <data_loading_example>`.
#
# We load example data from the lab dataset together with the INDIP reference
# system.

example_data = LabExampleDataset(
    reference_system="INDIP", reference_para_level="wb"
)

# %%
# Scope and assumptions
# ---------------------
# The algorithm groups possible gravity-alignment errors into four orientation
# families:
#
# - ``is_up``: gravity points up in sensor x (correct orientation).
# - ``is_down``: gravity points down in sensor x (180 deg rotation around
#   sensor z).
# - ``ml_up``: gravity points up in sensor y (90 deg rotation around sensor z).
# - ``ml_down``: gravity points down in sensor y (90 deg rotation around sensor
#   z, then 180 deg rotation around sensor x).
#
# Independently of the gravity direction, the sensor can also be flipped
# front-to-back around the vertical IS axis. The algorithm estimates this PA
# direction from the cross-spectral phase between ``acc_x`` and ``acc_z`` after
# gravity alignment.
#
# These assumptions are only valid for mostly upright walking bouts. Strongly
# hunched postures, non-walking activities, or very pathological gait patterns
# can make the reorientation fail.

# %%
# Performance on a single lab trial
# ---------------------------------
# Below we apply the algorithm to a lab trial, where we extract a single
# walking bout.
#
# The reorientation algorithm expects sensor-frame input and returns body-frame
# output.

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

first_wb = imu_data.loc[start:end].copy()

# %%
# Introducing artificial misorientation
# -------------------------------------
# To demonstrate the algorithm, we artificially introduce a misorientation.
# Below we rotate the sensor frame by 180 degrees around the sensor z-axis
# (``is_down``).

first_wb = flip_dataset(first_wb, Rotation.from_euler("z", 180, degrees=True))

print(first_wb)

# %%
# Visualising the misoriented walking bout
# ----------------------------------------
# We can visualise the walking bout before correction.

fig, ax = plt.subplots()

ax.plot(first_wb["acc_x"].to_numpy(), label="acc_x")
ax.plot(first_wb["acc_y"].to_numpy(), label="acc_y")
ax.plot(first_wb["acc_z"].to_numpy(), label="acc_z")

ax.legend()
fig.show()

# %%
# Applying the reorientation algorithm
# ------------------------------------
# Below we apply the ReorientationMethodDM algorithm to the misoriented walking
# bout.
# We use the 'full' correction mode which applies all three stages.

reoriented = ReorientationMethodDM(correction_mode="full").detect_correct(
    first_wb, sampling_rate_hz=sampling_rate_hz
)

print(f"\nDetected orientation family: {reoriented.result_.family}")
print(f"Correction applied: {reoriented.result_.correction_applied}")
print(f"Correction action: {reoriented.result_.correction_action}")

corrected = reoriented.corrected_data_

# %%
# Visualising the corrected walking bout
# --------------------------------------
# After correction, we can access the corrected data via the ``corrected_data_``
# attribute.

fig, ax = plt.subplots()

ax.plot(corrected["acc_is"].to_numpy(), label="acc_is")
ax.plot(corrected["acc_ml"].to_numpy(), label="acc_ml")
ax.plot(corrected["acc_pa"].to_numpy(), label="acc_pa")

ax.legend()
fig.show()


# %%
# Usage within the Mobilise-D pipeline
# --------------------------------------
# Instead of running the algorithm standalone, it can be used within the Mobilise-D pipeline to correct all gait
# sequences in a recording.
# The algorithm will be applied to each gait sequences detected by the used GSD algorithm.
#
# .. warning :: As the reorientation is performed after gait sequence detection, all algorithms before it must
#    be orientation-independent or explicitly support sensor-frame input.
#    This is NOT true for all GSD algorithms implemented in mobgap.
#    The pipeline trusts the configured algorithms and does not perform an additional frame-compatibility
#    check.
#
# Below we just show a simple example on how to do this, without artificial "flipping" of the data.

from mobgap.pipeline import MobilisedPipelineImpaired

pipeline = MobilisedPipelineImpaired(
    per_gs_reorientation=ReorientationMethodDM(correction_mode="full")
)

result = pipeline.run(single_test)
