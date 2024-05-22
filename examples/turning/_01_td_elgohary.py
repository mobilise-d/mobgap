"""
Elgohary Turning Algo
=========
This example shows how to use the Elgohary turning algorithm and some examples on how the results compare to the original
matlab implementation.
"""

from gaitmap.trajectory_reconstruction import MadgwickAHRS
from matplotlib import pyplot as plt

from mobgap.data import LabExampleDataset
from mobgap.turning import TdElGohary

# %%
# Loading data
# ------------
# .. note :: More infos about data loading can be found in the :ref:`data loading example <data_loading_example>`.
#
# We load example data from the lab dataset together with the INDIP reference system.
# We will use the Stereophoto output for turns ("td") as ground truth.
# Note, that the INDIP system, also uses just a single lower back IMU to calculate the turns.
# Hence, it can not really be considered a reference system, in this context.

example_data = LabExampleDataset(reference_system="Stereophoto", reference_para_level="wb")

single_test = example_data.get_subset(cohort="HA", participant_id="001", test="Test11", trial="Trial1")
imu_data = single_test.data_ss
reference_wbs = single_test.reference_parameters_.wb_list

sampling_rate_hz = single_test.sampling_rate_hz
ref_turns = single_test.reference_parameters_.turn_parameters

# %%
# Applying the algorithm
# ----------------------
# In a typical pipeline, we first identify the gait sequences and then apply the turning detection algorithm to each
# gait sequence individually.
# However, the turning algorithm can also be applied to the whole dataset at once.
# It might produce false positives, in "non-walking" segments.
#
# Below we show both approaches, starting with the whole dataset.
# This allows us to visualize how the algorithm works.
imu_data_rot = MadgwickAHRS().estimate(imu_data, sampling_rate_hz=sampling_rate_hz).rotated_data_


turning_detector = TdElGohary()

algo = turning_detector.detect(imu_data_rot, sampling_rate_hz=sampling_rate_hz)
turn_list = algo.turn_list_
turn_list

# %%
# We can also extract additional debug information from the algorithm.
# The yaw-angle gives us the estimated orientation of the lower back in the axis of the turning.
# The ``raw_turn_list_`` gives us the raw detected turns, before filtering them based on duration and angle.
yaw_angle = algo.yaw_angle_
raw_turns = algo.raw_turn_list_
raw_turns
# %%
# To better understand, how things work, we can plot all the results together.
#
# We can see that after filtering the signal, the algorithm identifies peaks in the signal that are higher in absolute
# values than the ``min_peak_angle_velocity_dps`` (dotted lines).
# Around these "turn-centers" the turn is defined as the region until the signal drops below the
# ``lower_threshold_velocity_dps`` (dashed lines).
#
# We then look at the duration and the angle of the detected turns and filter them based on the provided thresholds.
# We can see that at the end, only a small number of turns remain.
# Most of the raw turns are filtered out by the ``allowed_turn_angle_deg`` threshold, which is set to 45 degrees.
fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
axs[0].set_title("Raw gyr_z data")
axs[0].set_ylabel("gyr_z [dps]")
imu_data_rot.reset_index(drop=True).plot(y="gyr_z", ax=axs[0])

axs[1].set_title("Filtered IMU signal with raw turns and thresholds.")
axs[1].set_ylabel("filtered gyr_z [dps]")
filtered_data = (
    turning_detector.smoothing_filter.clone()
    .filter(imu_data_rot["gyr_z"], sampling_rate_hz=sampling_rate_hz)
    .filtered_data_
)
filtered_data.reset_index(drop=True).plot(ax=axs[1])
# Plot turn centeres
axs[1].plot(raw_turns["center"], filtered_data.iloc[raw_turns["center"]], "o")
# Plot start and end of turns as regions
for i, row in raw_turns.iterrows():
    axs[1].axvspan(row["start"], row["end"], alpha=0.5, color="gray" if row["direction"] == "left" else "blue")

# Draw dashed line at +/- velocity_dps
axs[1].axhline(turning_detector.lower_threshold_velocity_dps, color="green", linestyle="--", label="velocity_dps")
axs[1].axhline(-turning_detector.lower_threshold_velocity_dps, color="gray", linestyle="--")

# Draw dottet line at +/- height
axs[1].axhline(turning_detector.min_peak_angle_velocity_dps, color="green", linestyle=":", label="height")
axs[1].axhline(-turning_detector.min_peak_angle_velocity_dps, color="gray", linestyle=":")

axs[2].set_title("Yaw angle with final turns")
axs[2].set_ylabel("Yaw angle [deg]")
axs[2].set_xlabel("samples [#]")
yaw_angle.reset_index(drop=True).plot(ax=axs[2])

# Plot start and end of turns as regions
for i, row in turn_list.iterrows():
    axs[2].axvspan(row["start"], row["end"], alpha=0.5, color="gray" if row["direction"] == "left" else "blue")


fig.show()

# %%
# Now that we understand how the algorithm works, we apply it in the context of a typical pipeline using the
# ``GsIterator`` in combination with the reference WBs.
# This allows us to compare the results to the reference system.

from mobgap.pipeline import GsIterator

iterator = GsIterator()

imu_data_rot = MadgwickAHRS().estimate(imu_data, sampling_rate_hz=sampling_rate_hz).rotated_data_

for (gs, data), result in iterator.iterate(imu_data_rot, reference_wbs):
    td = turning_detector.clone().detect(data, sampling_rate_hz=sampling_rate_hz)
    result.turn_list = td.turn_list_

turns = iterator.results_.turn_list
turns

# %%
# We can compare this to the reference turns.
ref_turns

# %%
# Plotting the results
# --------------------
# imu_data.reset_index(drop=True).plot(y="gyr_x")
#
# plt.plot(ref_turns_frames["start"], imu_data.iloc[ref_turns_frames["start"]], "o", label="ref_start")
# plt.plot(ref_turns_frames["end"], imu_data.iloc[ref_turns_frames["end"]], "o", label="ref_end")
# plt.plot(detected_turns_py["td"], imu_data.iloc[detected_turns_py["td"]], "x", label="TD_Start_py")
# plt.plot(detected_end_py["td"], imu_data.iloc[detected_end_py["td"]], "x", label="TD_End_py")
# plt.plot(
#     detected_turns_matlab["TurnStart"], imu_data.iloc[detected_turns_matlab["TurnStart"]], "+", label="TD_Start_matlab"
# )
# plt.plot(detected_turns_matlab["TurnEnd"], imu_data.iloc[detected_turns_matlab["TurnEnd"]], "+", label="TD_End_matlab")
# plt.xlim(reference_wbs.iloc[5]["start"] - 50, reference_wbs.iloc[5]["end"] + 50)
# plt.legend()
# plt.show()

# With that we can compare the python, matlab and ground truth results.
# We zoom in into one of the gait sequences to better see the output.
#
# We can make a couple of main observations:
#
# 1. Python over-estimating the number of turns - need to check processing
# 2. WB 3 is perfect between all three
# 3. Python always detects turns 2 frames earlier than matlab
# 4. Turns are overestimated by both python and matlab in comparison to reference turns
# 5. difficult to directly compare with reference, due to turns being detected/postprocessing
# 6. Python and matlab are very well agreed on longer turns
# 7. shorter turns the python underestimates the turn angle in comparison to the matlab
# 8. Reference angle_deg of turns is much different in comparison to python and matlab
# 9. Both python and matlab detect more turns than the reference
