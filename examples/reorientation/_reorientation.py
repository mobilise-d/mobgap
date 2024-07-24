"""
Reorientation
=========

This example shows how to use the amended reorietation block and some examples on how the signal is corrected.

"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
# %%

# Loading data
# ------------
# .. note:: More infos about data loading can be found in the :ref:`data loading example <data_loading_example>`.
#
# We load the example data from the lab dataset together with 8 different datasets with wrong orientation
# based on the same example data.
from mobgap.data import LabExampleDataset

example_data = LabExampleDataset(reference_system="INDIP", reference_para_level="wb")
single_test = example_data.get_subset(cohort="HA", participant_id="001", test="Test11", trial="Trial1")
imu_data = single_test.data_ss
IMU_correct = imu_data


# %%
# Applying the algorithm example
# ----------------------
# Below we apply the reorientation block to a lab trial with correctly oriented data.
from mobgap.Reorientation._correct_orientation_sensor_axes import CorrectOrientationSensorAxes

reorientation = CorrectOrientationSensorAxes(sampling_rate_hz=100)
corrected = reorientation.update_orientation(IMU_correct)

corrected_IMU = corrected.corIMUdata
corrected_IMU
corrected_IMU_sequence = corrected.corIMUdataSequence
corrected_IMU_sequence


# TODO: load 8 wrong files, names could be IMU_*** based on the wrong orientation name


# %%

# Visualizing the signal before and after reorientation
# Below we visualize the signal before and after reorientation in the above file

fig, (pl1, pl2) = plt.subplots(2, 1, figsize=(10, 12))

# Pre reorientation
pl1.plot(corrected_IMU['index'], corrected_IMU['acc_x'], label='inferior-superior', color='green')
pl1.plot(corrected_IMU['index'], corrected_IMU['acc_y'], label='posterior-anterior', color='red')
pl1.plot(corrected_IMU['index'], corrected_IMU['acc_z'], label='medial-lateral', color='blue')
pl1.set_xlabel('Index')
pl1.set_ylabel('Acc')
pl1.set_title('Pre')
pl1.legend()

# Post reorientation
pl2.plot(corrected_IMU['index'], corrected_IMU['acc_x'], label='inferior-superior', color='green')
pl2.plot(corrected_IMU['index'], corrected_IMU['acc_y'], label='posterior-anterior', color='red')
pl2.plot(corrected_IMU['index'], corrected_IMU['acc_z'], label='medial-lateral', color='blue')
pl2.set_xlabel('Index')
pl2.set_ylabel('Acc')
pl2.set_title('Post')
pl2.legend()

plt.tight_layout()
plt.show()


# %%
# Applying the algorithm in wrongly oriented data
# ----------------------
# Below we apply the reorientation block to a lab trial with wrongly oriented data.
# The signal is rotated by 30 degrees on the ML axis to the opposite direction than the convention.
# TODO: add "Body Curvature"


# Visualizing the signal before and after reorientation
# Below we visualize the signal before and after reorientation in the above file
# TODO: ammend df names to reflect the pre and post reorientation

corrected_IMU = IMU_correct

fig, (pl1, pl2) = plt.subplots(2, 1, figsize=(10, 12))

# Pre reorientation
pl1.plot(corrected_IMU['index'], corrected_IMU['acc_x'], label='inferior-superior', color='green')
pl1.plot(corrected_IMU['index'], corrected_IMU['acc_y'], label='posterior-anterior', color='red')
pl1.plot(corrected_IMU['index'], corrected_IMU['acc_z'], label='medial-lateral', color='blue')
pl1.set_xlabel('Index')
pl1.set_ylabel('Acc')
pl1.set_title('Pre')
pl1.legend()

# Post reorientation
pl2.plot(corrected_IMU['index'], corrected_IMU['acc_x'], label='inferior-superior', color='green')
pl2.plot(corrected_IMU['index'], corrected_IMU['acc_y'], label='posterior-anterior', color='red')
pl2.plot(corrected_IMU['index'], corrected_IMU['acc_z'], label='medial-lateral', color='blue')
pl2.set_xlabel('Index')
pl2.set_ylabel('Acc')
pl2.set_title('Post')
pl2.legend()

plt.tight_layout()
plt.show()

# %%
# Applying the algorithm in wrongly oriented data
# ----------------------
# Below we apply the reorientation block to a lab trial with wrongly oriented data.
# The signal comes from a sensor placed upside downvertical and mediolateral axes will be inverted
# TODO: add "Upsidedown"

# Visualizing the signal before and after reorientation
# Below we visualize the signal before and after reorientation in the above file

# %%
# Applying the algorithm in wrongly oriented data
# ----------------------
# Below we apply the reorientation block to a lab trial with wrongly oriented data.
# The signal comes from a sensor placed inside out, mediolateral and anteroposterior are inverted.
# TODO: add "InsideOut"

# Visualizing the signal before and after reorientation
# Below we visualize the signal before and after reorientation in the above file

# %%
# Applying the algorithm in wrongly oriented data
# ----------------------
# Below we apply the reorientation block to a lab trial with wrongly oriented data.
# The signal comes from a sensor placed upside down and inside out, vertical and anteroposterior are inverted
# TODO: add "UpsidedownInsideOut"

# Visualizing the signal before and after reorientation
# Below we visualize the signal before and after reorientation in the above file

# %%
# Applying the algorithm in wrongly oriented data
# ----------------------
# Below we apply the reorientation block to a lab trial with wrongly oriented data.
# The signal comes from a sensor placed Clockwise with a 90-degree rotation
# TODO: add "C90"

# Visualizing the signal before and after reorientation
# Below we visualize the signal before and after reorientation in the above file

# %%
# Applying the algorithm in wrongly oriented data
# ----------------------
# Below we apply the reorientation block to a lab trial with wrongly oriented data.
# The signal comes from a sensor placed Clockwise with a 90-degree and inside out
# TODO: add "C90IO"

# Visualizing the signal before and after reorientation
# Below we visualize the signal before and after reorientation in the above file

# %%
# Applying the algorithm in wrongly oriented data
# ----------------------
# Below we apply the reorientation block to a lab trial with wrongly oriented data.
# The signal comes from a sensor placed counter-clockwise with a 90-degree rotation
# TODO: add "CC90"

# Visualizing the signal before and after reorientation
# Below we visualize the signal before and after reorientation in the above file

# %%
# Applying the algorithm in wrongly oriented data
# ----------------------
# Below we apply the reorientation block to a lab trial with wrongly oriented data.
# The signal comes from a sensor placed counter-clockwise with a 90-degree rotation and inside out
# TODO: add "CC90IO"

# Visualizing the signal before and after reorientation
# Below we visualize the signal before and after reorientation in the above file
