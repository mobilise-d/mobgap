"""
ElGohary Turning Algo
=====================

.. warning:: There are some issues with matching the results of the ElGohary algorithm to the reference system.
             The performance, we are observing here is far below the expected performance.
             We are investigating this currently, but until then, we recommend to do your own testing and validation
             before using this algorithm in production.

This example shows how to use the ElGohary turning algorithm.
It uses the angular velocity around the SI axis of a lower back IMU to detect turns.
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

print(single_test.participant_metadata)

sampling_rate_hz = single_test.sampling_rate_hz
ref_turns = single_test.reference_parameters_.turn_parameters

# %%
# Applying the algorithm
# ----------------------
# In a typical pipeline, we first identify the gait sequences and then apply the turning detection algorithm to each
# gait sequence individually.
# However, the turning algorithm can also be applied to the whole recording at once.
# Though, it might produce false positives, in "non-walking" segments.
#
# Below we show both approaches, starting with the whole recording.
# This allows us to visualize how the algorithm works.
turning_detector = TdElGohary()

turning_detector.detect(imu_data, sampling_rate_hz=sampling_rate_hz)
turn_list = turning_detector.turn_list_
turn_list

# %%
# We can also extract additional debug information from the algorithm.
# The yaw-angle gives us the estimated orientation of the lower back in the axis of the turning.
# The ``raw_turn_list_`` gives us the raw detected turns, before filtering them based on duration and angle.
yaw_angle = turning_detector.yaw_angle_
raw_turns = turning_detector.raw_turn_list_
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
def plot_turns(algo_with_results: TdElGohary):
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axs[0].set_title("Raw gyr_z data")
    axs[0].set_ylabel("gyr_z [dps]")
    algo_with_results.data.reset_index(drop=True).plot(y="gyr_z", ax=axs[0])

    axs[1].set_title("Filtered IMU signal with raw turns and thresholds.")
    axs[1].set_ylabel("filtered gyr_z [dps]")
    filtered_data = (
        algo_with_results.smoothing_filter.clone()
        .filter(imu_data["gyr_z"], sampling_rate_hz=sampling_rate_hz)
        .filtered_data_
    )
    filtered_data.reset_index(drop=True).plot(ax=axs[1])

    raw_turn_list = algo_with_results.raw_turn_list_
    # Plot turn centeres
    axs[1].plot(raw_turn_list["center"], filtered_data.iloc[raw_turn_list["center"]], "o")
    # Plot start and end of turns as regions
    for i, row in raw_turn_list.iterrows():
        axs[1].axvspan(row["start"], row["end"], alpha=0.5, color="gray" if row["direction"] == "left" else "blue")

    # Draw dashed line at +/- velocity_dps
    axs[1].axhline(algo_with_results.lower_threshold_velocity_dps, color="green", linestyle="--", label="velocity_dps")
    axs[1].axhline(-algo_with_results.lower_threshold_velocity_dps, color="gray", linestyle="--")

    # Draw dottet line at +/- height
    axs[1].axhline(algo_with_results.min_peak_angle_velocity_dps, color="green", linestyle=":", label="height")
    axs[1].axhline(-algo_with_results.min_peak_angle_velocity_dps, color="gray", linestyle=":")

    axs[2].set_title("Yaw angle with final turns")
    axs[2].set_ylabel("Yaw angle [deg]")
    axs[2].set_xlabel("samples [#]")
    algo_with_results.yaw_angle_.reset_index(drop=True).plot(ax=axs[2])

    # Plot start and end of turns as regions
    for i, row in algo_with_results.turn_list_.iterrows():
        axs[2].axvspan(row["start"], row["end"], alpha=0.5, color="gray" if row["direction"] == "left" else "blue")

    fig.show()

plot_turns(turning_detector)

# %%
# Now that we understand how the algorithm works, we apply it in the context of a typical pipeline using the
# ``GsIterator`` in combination with the reference WBs.
# This allows us to compare the results to the reference system.
from mobgap.pipeline import GsIterator

iterator = GsIterator()

imu_data = MadgwickAHRS().estimate(imu_data, sampling_rate_hz=sampling_rate_hz).rotated_data_

for (gs, data), result in iterator.iterate(imu_data, reference_wbs):
    td = turning_detector.clone().detect(data, sampling_rate_hz=sampling_rate_hz)
    result.turn_list = td.turn_list_

turns = iterator.results_.turn_list
turns

# %%
# We can compare this to the reference turns.
ref_turns

# %%
# We can directly observe that the algorithm detects significantly fewer turns than the reference system.
# And even the turns that are detected, don't really match the reference system.
#
# It remains unclear, why the algorithm performs so poorly in this case.

# %%
# Working in the global coordinate system
# ---------------------------------------
# The original ElGohary paper uses on-board sensor fusion to track the orientation of the sensor.
# This information is used to transform all sensor data into the global coordinate system.
# This should make the identification of the SI axis more robust.
#
# We did not use this approach within Mobilise-D, to avoid introducing an additional source of error through a sensor
# fusion algorithm.
# However, the result of the algorithms are significantly influenced by that decision.
# Below we show how you could use the algorithm with a prior global frame estimation.
#
# For this we are using the MadgwickAHRS algorithm to estimate the global orientation of the sensor and transform the
# data into the global coordinate system.
from gaitmap.trajectory_reconstruction import MadgwickAHRS

imu_data = MadgwickAHRS().estimate(imu_data, sampling_rate_hz=sampling_rate_hz).rotated_data_

# %%
# Now we can apply the algorithm again.
# We are going to apply it to the entire recording to show the step-by-step process.
turning_detector = TdElGohary()

turning_detector.detect(imu_data, sampling_rate_hz=sampling_rate_hz)
plot_turns(turning_detector)

# %%
# Compared to the previous results, we can see that the algorithm now detects significantly more turns.
# However, this does not necessarily mean that the results are better, nor that this finding is consistent across all
# recordings.
