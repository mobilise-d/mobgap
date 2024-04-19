"""
Elgohary Turning Algo
=========
This example shows how to use the Elgohary turning algorithm and some examples on how the results compare to the original
matlab implementation.
"""

import pandas as pd
from gaitmap.trajectory_reconstruction.orientation_methods import MadgwickAHRS
from gaitmap.utils.rotations import rotate_dataset_series
from matplotlib import pyplot as plt

from mobgap.data import LabExampleDataset
from mobgap.turn_detection import TdElgohary

# %%
# Loading data
# ------------
# .. note :: More infos about data loading can be found in the :ref:`data loading example <data_loading_example>`.
# We load example data from the lab dataset together with the INDIP reference system.
# We will use the INDIP output for turns ("td") as ground truth.

example_data = LabExampleDataset(reference_system="Stereophoto", reference_para_level="wb")

single_test = example_data.get_subset(cohort="HA", participant_id="001", test="Test11", trial="Trial1")
imu_data = single_test.data_ss
reference_wbs = single_test.reference_parameters_.wb_list

sampling_rate_hz = single_test.sampling_rate_hz
ref_turns = single_test.reference_parameters_.turn_parameters

turning_detector = TdElgohary()


reference_wbs

# %%
# Applying the algorithm
# ----------------------
# Below we apply the shin algorithm to a lab trial.
# We will use the `GsIterator` to iterate over the gait sequences and apply the algorithm to each wb.

from mobgap.pipeline import GsIterator

iterator = GsIterator()

single_wb = reference_wbs.loc[3]
single_wb_data = imu_data.iloc[single_wb["start"] : single_wb["end"]]


# single_wb_data = rotate_dataset_series(
#     single_wb_data, MadgwickAHRS().estimate(single_wb_data, sampling_rate_hz=sampling_rate_hz).orientation_object_[:-1]
# )

algo = turning_detector.detect(single_wb_data, sampling_rate_hz=sampling_rate_hz)
yaw_angle = algo.yaw_angle_
raw_turns = algo.raw_turns_

print(raw_turns)

# Plot gyr_z data
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
single_wb_data.reset_index(drop=True).plot(y="gyr_z", ax=axs[0])
filtered_data = (
    turning_detector.smoothing_filter.clone()
    .filter(single_wb_data["gyr_z"], sampling_rate_hz=sampling_rate_hz)
    .filtered_data_
)
pd.Series(filtered_data).reset_index(drop=True).plot(ax=axs[1])
# Plot turn centeres
axs[1].plot(raw_turns["center"], filtered_data.iloc[raw_turns["center"]], "o", label="Turn_Start")
# Plot start and end of turns as regions
for i, row in raw_turns.iterrows():
    axs[1].axvspan(row["start"], row["end"], alpha=0.5, color="gray" if row["direction"] == "left" else "green")

# Draw dashed line at +/- velocity_dps
axs[1].axhline(turning_detector.velocity_dps, color="green", linestyle="--", label="velocity_dps")
axs[1].axhline(-turning_detector.velocity_dps, color="gray", linestyle="--")

# Draw dottet line at +/- height
axs[1].axhline(turning_detector.height, color="green", linestyle=":", label="height")
axs[1].axhline(-turning_detector.height, color="gray", linestyle=":")

fig.show()


for (gs, data), result in iterator.iterate(imu_data, reference_wbs):
    # Store the turns info onto results list
    td = turning_detector.clone().detect(data, sampling_rate_hz=sampling_rate_hz)
    result.turn_list = td.turn_list_

turns = iterator.results_.turn_list
turns

# %%
# Matlab Outputs
# --------------
# To check if the algorithm was implemented correctly, we compare the results to the matlab implementation.
import json

from mobgap import PACKAGE_ROOT


def load_matlab_output(datapoint):
    p = datapoint.group_label
    with (
        PACKAGE_ROOT.parent
        / f"example_data/original_results/td_elgohary/lab/{p.cohort}/{p.participant_id}/TD_Output.json"
    ).open() as f:
        original_results = json.load(f)["TD_Output"][p.time_measure][p.test][p.trial]["SU"]["LowerBack"]["TD"]

    if not isinstance(original_results, list):
        original_results = [original_results]

    turns = {}
    turns_end = {}
    turns_duration = {}
    turns_angle = {}
    for i, gs in enumerate(original_results):
        turns[i] = pd.DataFrame({"TurnStart": [gs["Turn_Start"]]}, index=["td_id"]).rename_axis(index="td_id")
        turns_end[i] = pd.DataFrame({"TurnEnd": [gs["Turn_End"]]}, index=["td_id"]).rename_axis(index="td_id")
        turns_duration[i] = pd.DataFrame({"TurnDuration": [gs["Turn_Duration"]]}, index=["td_id"]).rename_axis(
            index="td_id"
        )
        turns_angle[i] = pd.DataFrame({"Turn_Angle": [gs["Turn_Angle"]]}, index=["td_id"]).rename_axis(index="td_id")

    return (
        pd.concat(
            [
                pd.concat(turns, names=["wb_id", turns[0].index.name]) * datapoint.sampling_rate_hz,
                pd.concat(turns_end, names=["wb_id", turns_end[0].index.name]) * datapoint.sampling_rate_hz,
                pd.concat(turns_duration, names=["wb_id", turns_duration[0].index.name]) * datapoint.sampling_rate_hz,
                pd.concat(turns_angle, names=["wb_id", turns_angle[0].index.name]),
            ],
            axis=1,
        )
    ).astype(int)


detected_turns_matlab = load_matlab_output(single_test)
detected_turns_matlab

# %%
# Plotting the results
# --------------------
imu_data.reset_index(drop=True).plot(y="gyr_x")

plt.plot(ref_turns_frames["start"], imu_data.iloc[ref_turns_frames["start"]], "o", label="ref_start")
plt.plot(ref_turns_frames["end"], imu_data.iloc[ref_turns_frames["end"]], "o", label="ref_end")
plt.plot(detected_turns_py["td"], imu_data.iloc[detected_turns_py["td"]], "x", label="TD_Start_py")
plt.plot(detected_end_py["td"], imu_data.iloc[detected_end_py["td"]], "x", label="TD_End_py")
plt.plot(
    detected_turns_matlab["TurnStart"], imu_data.iloc[detected_turns_matlab["TurnStart"]], "+", label="TD_Start_matlab"
)
plt.plot(detected_turns_matlab["TurnEnd"], imu_data.iloc[detected_turns_matlab["TurnEnd"]], "+", label="TD_End_matlab")
plt.xlim(reference_wbs.iloc[5]["start"] - 50, reference_wbs.iloc[5]["end"] + 50)
plt.legend()
plt.show()

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
