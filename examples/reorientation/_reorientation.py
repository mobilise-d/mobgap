"""
Reorientation
=========

This example shows how to use the amended reorientation block and some examples on how the signal is corrected.

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
from mobgap.utils.conversions import to_body_frame

example_data = LabExampleDataset(reference_system="INDIP", reference_para_level="wb")
single_test = example_data.get_subset(cohort="HA", participant_id="001", test="Test11", trial="Trial1")
IMU = to_body_frame(single_test.data_ss)

# %%
# Applying the algorithm example
# ----------------------
# Below we apply the reorientation block to a lab trial with correctly oriented data.
from mobgap.Reorientation._correct_orientation_sensor_axes import CorrectOrientationSensorAxes

reorientation = CorrectOrientationSensorAxes(sampling_rate_hz=100)
corrected = reorientation.update_orientation(IMU)

corrected_IMU = corrected.corIMUdata
corrected_IMU
corrected_IMU_sequence = corrected.corIMUdataSequence
corrected_IMU_sequence

# %%

# Visualizing the signal before and after reorientation
# Below we visualize the signal before and after reorientation in the above file

fig, (pl1, pl2) = plt.subplots(2, 1, figsize=(10, 12))

# Pre reorientation
pl1.plot(IMU.index, IMU['acc_is'], label='inferior-superior', color='green')
pl1.plot(IMU.index, IMU['acc_ml'], label='medial-lateral', color='red')
pl1.plot(IMU.index, IMU['acc_pa'], label='posterior-anterior', color='blue')
pl1.set_xlabel('Index')
pl1.set_ylabel('Acc')
pl1.set_title('Pre')
pl1.legend()

# Post reorientation
pl2.plot(corrected_IMU.index, corrected_IMU['acc_is'], label='inferior-superior', color='green')
pl2.plot(corrected_IMU.index, corrected_IMU['acc_ml'], label='medial-lateral', color='red')
pl2.plot(corrected_IMU.index, corrected_IMU['acc_pa'], label='posterior-anterior', color='blue')
pl2.set_xlabel('Index')
pl2.set_ylabel('Acc')
pl2.set_title('Post')
pl2.legend()

plt.tight_layout()
plt.savefig('/Users/dimitrismegaritis/Desktop/correctorientation.png')

plt.show()



# %%
# Applying the algorithm in wrongly oriented data
# ----------------------
# Below we apply the reorientation block to a lab trial with wrongly oriented data.
# The signal is rotated by 30 degrees on the ML axis to the opposite direction than the convention.
# After the re-orientation it looks like the reorientation does not work as expected in this kind of signal.
# Within walking bouts, the inferior-superior (vertical) axis is reduced close to 0 instead increasing close to 10 m/s^2.

# Loading the data
path = 'example_data/data_csv/data_reorientation/BodyCurvature.csv'
IMU_BD = pd.read_csv(path)

reorientation = CorrectOrientationSensorAxes(sampling_rate_hz=100)
corrected = reorientation.update_orientation(IMU_BD)

corrected_IMU = corrected.corIMUdata
corrected_IMU
corrected_IMU_sequence = corrected.corIMUdataSequence
corrected_IMU_sequence

# Visualizing the signal before and after reorientation
# Below we visualize the signal before and after reorientation in the above file
fig, (pl1, pl2) = plt.subplots(2, 1, figsize=(10, 12))

# Pre reorientation
pl1.plot(IMU_BD.index, IMU_BD['acc_is'], label='inferior-superior', color='green')
pl1.plot(IMU_BD.index, IMU_BD['acc_ml'], label='medial-lateral', color='red')
pl1.plot(IMU_BD.index, IMU_BD['acc_pa'], label='posterior-anterior', color='blue')
pl1.set_xlabel('Index')
pl1.set_ylabel('Acc')
pl1.set_title('Pre')
pl1.legend()

# Post reorientation
pl2.plot(corrected_IMU.index, corrected_IMU['acc_is'], label='inferior-superior', color='green')
pl2.plot(corrected_IMU.index, corrected_IMU['acc_ml'], label='medial-lateral', color='red')
pl2.plot(corrected_IMU.index, corrected_IMU['acc_pa'], label='posterior-anterior', color='blue')
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
# The signal comes from a sensor placed upside down, so inferior-superior (vertical) and mediolateral axes will be inverted
# After the reorientation inferior-superior (vertical) is corrected but mediolateral remains the same

# Loading the data for Upsidedown
path = 'example_data/data_csv/data_reorientation/Upsidedown.csv'  # Adjust the file path
IMU_Upsidedown = pd.read_csv(path)

# Assuming CorrectOrientationSensorAxes is defined elsewhere in your code
reorientation = CorrectOrientationSensorAxes(sampling_rate_hz=100)
corrected = reorientation.update_orientation(IMU_Upsidedown)

corrected_IMU = corrected.corIMUdata
corrected_IMU_sequence = corrected.corIMUdataSequence

# Visualizing the signal before and after reorientation
fig, (pl1, pl2) = plt.subplots(2, 1, figsize=(10, 12))

# Pre reorientation
pl1.plot(IMU_Upsidedown.index, IMU_Upsidedown['acc_is'], label='inferior-superior', color='green')
pl1.plot(IMU_Upsidedown.index, IMU_Upsidedown['acc_ml'], label='medial-lateral', color='red')
pl1.plot(IMU_Upsidedown.index, IMU_Upsidedown['acc_pa'], label='posterior-anterior', color='blue')
pl1.set_xlabel('Index')
pl1.set_ylabel('Acc')
pl1.set_title('Pre')
pl1.legend()

# Post reorientation
pl2.plot(corrected_IMU.index, corrected_IMU['acc_is'], label='inferior-superior', color='green')
pl2.plot(corrected_IMU.index, corrected_IMU['acc_ml'], label='medial-lateral', color='red')
pl2.plot(corrected_IMU.index, corrected_IMU['acc_pa'], label='posterior-anterior', color='blue')
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
# The signal comes from a sensor placed inside out, so mediolateral and anteroposterior are inverted compared to the correct orientation.
# After the reorientation, mediolateral and anteroposterior remain the same

# Loading the data for InsideOut
path = 'example_data/data_csv/data_reorientation/InsideOut.csv'  # Adjust the file path
IMU_InsideOut = pd.read_csv(path)

# Assuming CorrectOrientationSensorAxes is defined elsewhere in your code
reorientation = CorrectOrientationSensorAxes(sampling_rate_hz=100)
corrected = reorientation.update_orientation(IMU_InsideOut)

corrected_IMU = corrected.corIMUdata
corrected_IMU_sequence = corrected.corIMUdataSequence

# Visualizing the signal before and after reorientation
fig, (pl1, pl2) = plt.subplots(2, 1, figsize=(10, 12))

# Pre reorientation
pl1.plot(IMU_InsideOut.index, IMU_InsideOut['acc_is'], label='inferior-superior', color='green')
pl1.plot(IMU_InsideOut.index, IMU_InsideOut['acc_ml'], label='medial-lateral', color='red')
pl1.plot(IMU_InsideOut.index, IMU_InsideOut['acc_pa'], label='posterior-anterior', color='blue')
pl1.set_xlabel('Index')
pl1.set_ylabel('Acc')
pl1.set_title('Pre')
pl1.legend()

# Post reorientation
pl2.plot(corrected_IMU.index, corrected_IMU['acc_is'], label='inferior-superior', color='green')
pl2.plot(corrected_IMU.index, corrected_IMU['acc_ml'], label='medial-lateral', color='red')
pl2.plot(corrected_IMU.index, corrected_IMU['acc_pa'], label='posterior-anterior', color='blue')
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
# The signal comes from a sensor placed upside down and inside out, so the inferior-superior (vertical) and anteroposterior
# are inverted compared to the correct orientation
# After the reorientation, vertical is inverted but anteroposterior remains the same

# Loading the data for UpsidedownInsideOut
path = 'example_data/data_csv/data_reorientation/UpsidedownInsideOut.csv'  # Adjust the file path
IMU_UpsidedownInsideOut = pd.read_csv(path)

# Assuming CorrectOrientationSensorAxes is defined elsewhere in your code
reorientation = CorrectOrientationSensorAxes(sampling_rate_hz=100)
corrected = reorientation.update_orientation(IMU_UpsidedownInsideOut)

corrected_IMU = corrected.corIMUdata
corrected_IMU_sequence = corrected.corIMUdataSequence

# Visualizing the signal before and after reorientation
fig, (pl1, pl2) = plt.subplots(2, 1, figsize=(10, 12))

# Pre reorientation
pl1.plot(IMU_UpsidedownInsideOut.index, IMU_UpsidedownInsideOut['acc_is'], label='inferior-superior', color='green')
pl1.plot(IMU_UpsidedownInsideOut.index, IMU_UpsidedownInsideOut['acc_ml'], label='medial-lateral', color='red')
pl1.plot(IMU_UpsidedownInsideOut.index, IMU_UpsidedownInsideOut['acc_pa'], label='posterior-anterior', color='blue')
pl1.set_xlabel('Index')
pl1.set_ylabel('Acc')
pl1.set_title('Pre')
pl1.legend()

# Post reorientation
pl2.plot(corrected_IMU.index, corrected_IMU['acc_is'], label='inferior-superior', color='green')
pl2.plot(corrected_IMU.index, corrected_IMU['acc_ml'], label='medial-lateral', color='red')
pl2.plot(corrected_IMU.index, corrected_IMU['acc_pa'], label='posterior-anterior', color='blue')
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
# The signal comes from a sensor rotated clockwise by 90-degrees so infererior-superior (vertical) is equal to the mediolateral
# of the correct orientation and the mediolateral is equal to the opposite of the inferior-superior (vertical) of
# the correct orientation
# After the re-orientation only within the walking bouts, some corrections are made to the mediolateral axis

# Loading the data for C90
path = 'example_data/data_csv/data_reorientation/C90.csv'
IMU_C90 = pd.read_csv(path)

# Assuming CorrectOrientationSensorAxes is defined elsewhere in your code
reorientation = CorrectOrientationSensorAxes(sampling_rate_hz=100)
corrected = reorientation.update_orientation(IMU_C90)

corrected_IMU = corrected.corIMUdata
corrected_IMU_sequence = corrected.corIMUdataSequence

# Visualizing the signal before and after reorientation
fig, (pl1, pl2) = plt.subplots(2, 1, figsize=(10, 12))

# Pre reorientation
pl1.plot(IMU_C90.index, IMU_C90['acc_is'], label='inferior-superior', color='green')
pl1.plot(IMU_C90.index, IMU_C90['acc_ml'], label='medial-lateral', color='red')
pl1.plot(IMU_C90.index, IMU_C90['acc_pa'], label='posterior-anterior', color='blue')
pl1.set_xlabel('Index')
pl1.set_ylabel('Acc')
pl1.set_title('Pre')
pl1.legend()

# Post reorientation
pl2.plot(corrected_IMU.index, corrected_IMU['acc_is'], label='inferior-superior', color='green')
pl2.plot(corrected_IMU.index, corrected_IMU['acc_ml'], label='medial-lateral', color='red')
pl2.plot(corrected_IMU.index, corrected_IMU['acc_pa'], label='posterior-anterior', color='blue')
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
# The signal comes from a sensor rotated clockwise by 90-degrees and "inside out" so anteroposterior is reversed
# compared to the correct orientation, mediolateral is similar to inferior-superior but reversed compared to the correct
# orientation, inferior-superior is similar to mediolateral
# After the re-orientation only within the walking bouts, some corrections are made to the mediolateral axis

# Loading the data for C90IO
path = 'example_data/data_csv/data_reorientation/C90IO.csv'
IMU_C90IO = pd.read_csv(path)

# Assuming CorrectOrientationSensorAxes is defined elsewhere in your code
reorientation = CorrectOrientationSensorAxes(sampling_rate_hz=100)
corrected = reorientation.update_orientation(IMU_C90IO)

corrected_IMU = corrected.corIMUdata
corrected_IMU_sequence = corrected.corIMUdataSequence

# Visualizing the signal before and after reorientation
fig, (pl1, pl2) = plt.subplots(2, 1, figsize=(10, 12))

# Pre reorientation
pl1.plot(IMU_C90IO.index, IMU_C90IO['acc_is'], label='inferior-superior', color='green')
pl1.plot(IMU_C90IO.index, IMU_C90IO['acc_ml'], label='medial-lateral', color='red')
pl1.plot(IMU_C90IO.index, IMU_C90IO['acc_pa'], label='posterior-anterior', color='blue')
pl1.set_xlabel('Index')
pl1.set_ylabel('Acc')
pl1.set_title('Pre')
pl1.legend()

# Post reorientation
pl2.plot(corrected_IMU.index, corrected_IMU['acc_is'], label='inferior-superior', color='green')
pl2.plot(corrected_IMU.index, corrected_IMU['acc_ml'], label='medial-lateral', color='red')
pl2.plot(corrected_IMU.index, corrected_IMU['acc_pa'], label='posterior-anterior', color='blue')
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
# The signal comes from a sensor placed rotated counter-clockwise by 90-degrees so mediolateral is similar to the
# inferior-superior (vertical) of the correct orientation and inferior-superior (vertical) is the opposite of
# mediolateral of the correct orientation
# After the re-orientation only within the walking bouts, some corrections are made only to the mediolateral axis

# Loading the data for CC90
path = 'example_data/data_csv/data_reorientation/CC90.csv'
IMU_CC90 = pd.read_csv(path)

# Assuming CorrectOrientationSensorAxes is defined elsewhere in your code
reorientation = CorrectOrientationSensorAxes(sampling_rate_hz=100)
corrected = reorientation.update_orientation(IMU_CC90)

corrected_IMU = corrected.corIMUdata
corrected_IMU_sequence = corrected.corIMUdataSequence

# Visualizing the signal before and after reorientation
fig, (pl1, pl2) = plt.subplots(2, 1, figsize=(10, 12))

# Pre reorientation
pl1.plot(IMU_CC90.index, IMU_CC90['acc_is'], label='inferior-superior', color='green')
pl1.plot(IMU_CC90.index, IMU_CC90['acc_ml'], label='medial-lateral', color='red')
pl1.plot(IMU_CC90.index, IMU_CC90['acc_pa'], label='posterior-anterior', color='blue')
pl1.set_xlabel('Index')
pl1.set_ylabel('Acc')
pl1.set_title('Pre')
pl1.legend()

# Post reorientation
pl2.plot(corrected_IMU.index, corrected_IMU['acc_is'], label='inferior-superior', color='green')
pl2.plot(corrected_IMU.index, corrected_IMU['acc_ml'], label='medial-lateral', color='red')
pl2.plot(corrected_IMU.index, corrected_IMU['acc_pa'], label='posterior-anterior', color='blue')
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
# The signal comes from a sensor rotated counter-clockwise by 90-degrees and inside out so inferior-superior (vertical)
# is similar to mediolateral of the correct orientation, mediolateral is similar to inferior-superior (vertical) of
# the correct orientation, and anterior-posterior is reversed
# After the re-orientation only within the walking bouts, some corrections are made only to the mediolateral axis

# Loading the data for CC90IO
path = 'example_data/data_csv/data_reorientation/CC90IO.csv'
IMU_CC90IO = pd.read_csv(path)

# Assuming CorrectOrientationSensorAxes is defined elsewhere in your code
reorientation = CorrectOrientationSensorAxes(sampling_rate_hz=100)
corrected = reorientation.update_orientation(IMU_CC90IO)

corrected_IMU = corrected.corIMUdata
corrected_IMU_sequence = corrected.corIMUdataSequence

# Visualizing the signal before and after reorientation
fig, (pl1, pl2) = plt.subplots(2, 1, figsize=(10, 12))

# Pre reorientation
pl1.plot(IMU_CC90IO.index, IMU_CC90IO['acc_is'], label='inferior-superior', color='green')
pl1.plot(IMU_CC90IO.index, IMU_CC90IO['acc_ml'], label='medial-lateral', color='red')
pl1.plot(IMU_CC90IO.index, IMU_CC90IO['acc_pa'], label='posterior-anterior', color='blue')
pl1.set_xlabel('Index')
pl1.set_ylabel('Acc')
pl1.set_title('Pre')
pl1.legend()

# Post reorientation
pl2.plot(corrected_IMU.index, corrected_IMU['acc_is'], label='inferior-superior', color='green')
pl2.plot(corrected_IMU.index, corrected_IMU['acc_ml'], label='medial-lateral', color='red')
pl2.plot(corrected_IMU.index, corrected_IMU['acc_pa'], label='posterior-anterior', color='blue')
pl2.set_xlabel('Index')
pl2.set_ylabel('Acc')
pl2.set_title('Post')
pl2.legend()

plt.tight_layout()
plt.show()
