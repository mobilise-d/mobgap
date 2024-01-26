"""
ICD Ionescu
========.

This example shows how to use the ICD Ionescu algorithm and how its results compare to the original
matlab implementation.

"""

# %%
# Import useful modules and packages
import matplotlib.pyplot as plt
import pandas as pd

from gaitlink.data import LabExampleDataset
from gaitlink.icd import IcdIonescu

# %%
# Loading some example data
# -------------------------
# .. note :: More infos about data loading can be found in the :ref:`data loading example <data_loading_example>`.
#
# We load example data from the lab dataset together with the INDIP reference system.
# We will use the INDIP "InitialContact_Event" output as ground truth.

example_data = LabExampleDataset(reference_system="INDIP", reference_para_level="wb")  # alternatively: "StereoPhoto"

# %%
# Performance on a single lab trial
# ---------------------------------
# Below we apply the algorithm to a lab trail, where we only expect a single gait sequence.
single_test = example_data.get_subset(cohort="HA", participant_id="001", test="Test11", trial="Trial1")
imu_data = single_test.data["LowerBack"]
reference_wbs = single_test.reference_parameters_.walking_bouts

sampling_rate_hz = single_test.sampling_rate_hz
ref_ics = single_test.reference_parameters_.initial_contacts

reference_wbs

# %%
# Applying the algorithm
# ----------------------
# Below we apply the shin algorithm to a lab trial.
# We will use the `GsIterator` to iterate over the gait sequences and apply the algorithm to each wb.
from gaitlink.pipeline import GsIterator

iterator = GsIterator()

for (gs, data), result in iterator.iterate(imu_data, reference_wbs):
    result.initial_contacts = IcdIonescu().detect(data, sampling_rate_hz=sampling_rate_hz).ic_list_

detected_ics = iterator.initial_contacts_
detected_ics

# %%
# Matlab Outputs
# --------------
# To check if the algorithm was implemented correctly, we compare the results to the matlab implementation.
import json

from gaitlink import PACKAGE_ROOT


def load_matlab_output(datapoint):
    p = datapoint.group_label
    with (
        PACKAGE_ROOT.parent
        / f"example_data/original_results/icd_ionescu/lab/{p.cohort}/{p.participant_id}/SD_Output.json"
    ).open() as f:
        original_results = json.load(f)["SD_Output"][p.time_measure][p.test][p.trial]["SU"]["LowerBack"]["SD"]

    if not isinstance(original_results, list):
        original_results = [original_results]

    ics = {}
    for i, gs in enumerate(original_results):
        ics[i] = pd.DataFrame({"ic": gs["IC"]}).rename_axis(index="ic_id")

    return (pd.concat(ics, names=["wb_id", ics[0].index.name]) * datapoint.sampling_rate_hz).astype(int)


detected_ics_matlab = load_matlab_output(single_test)
detected_ics_matlab
# %%
# Plotting the results
# --------------------

imu_data.reset_index(drop=True).plot(y="acc_x")

plt.plot(ref_ics["ic"], imu_data["acc_x"].iloc[ref_ics["ic"]], "o", label="ref")
plt.plot(detected_ics["ic"], imu_data["acc_x"].iloc[detected_ics["ic"]], "x", label="icd_ionescu_py")
plt.plot(detected_ics_matlab["ic"], imu_data["acc_x"].iloc[detected_ics_matlab["ic"]], "+", label="icd_ionescu_matlab")
plt.xlim(reference_wbs.iloc[2]["start"] - 50, reference_wbs.iloc[2]["end"] + 50)
plt.legend()
plt.show()


# fs = single_test.sampling_rate_hz  # sampling rate (Hz)
# # if the first sample is t = 0 s --> we don't need to subtract -1 to the gait sequenced
# # TODO: Update to new format
# # TODO: Check start and end of gait sequence -1 required or not?
# s = round(single_test.reference_parameters_["lwb"][0]["Start"] * fs) - 1  # start of gait sequence (samples)
# e = round(single_test.reference_parameters_["lwb"][0]["End"] * fs) - 1  # end of gait sequence (samples)
# gs = single_test.data["LowerBack"][s : e + 1]  # imu data during gait sequence
# single_test_output = IcdIonescu().detect(gs, sampling_rate_hz=fs)  # output of ICDs
# ICs_from_start = (
#     single_test.reference_parameters_["lwb"][0]["Start"] + single_test_output.icd_list_
# )  # ICs (s) from the start of the Trial
# print("Reference Parameters:\n\n", single_test_reference_parameters)
# print("\nMatlab Output:\n\n", single_test_matlab_output)
# print("\nPython Output:\n\n", single_test_output.icd_list_)
#
# # %% Plot and comparison with Matlab
# IC_matlab_seconds = [x for x in single_test_matlab_output["ics"][0]]
# IC_matlab_samples = [round(x * fs) for x in single_test_matlab_output["ics"][0]]
# IC_INDIP_seconds = [x / fs for x in single_test_reference_parameters["ICs"][0]]
# IC_INDIP_samples = [round(x) for x in single_test_reference_parameters["ICs"][0]]
#
# # do two pics--> one general and one more zoomed-in
# accV = single_test.data["LowerBack"]["acc_x"]
# plt.close()
# fig, ax = plt.subplots()
# t = np.arange(1 / fs, (len(accV) + 1) / fs, 1 / fs, dtype=float)
# ax.plot(t, accV)
# ax.plot(ICs_from_start, accV.array[(ICs_from_start * fs).astype(int)].to_numpy(), "ro", label="Python")
# ax.plot(IC_matlab_seconds, accV.array[IC_matlab_samples], "b*", label="Matlab")
# ax.plot(IC_INDIP_seconds, accV.array[IC_INDIP_samples], "k+", label="INDIP")
# ax.fill_betweenx(np.arange(min(accV) - 1, max(accV) + 1, 0.01), s / fs, e / fs, facecolor="green", alpha=0.2)
# plt.xlabel("Time (s)")
# plt.ylabel("Vertical Acceleration (m/s^2)")
# plt.title("IC detection: HA002 - Test 5 - Trial 2")
# plt.legend(loc="upper left")
# plt.show()

# %%
# When we plot the output, we can see that both algorithm implementations ignore
# the ICs given by the first and the last element of the gait sequence
