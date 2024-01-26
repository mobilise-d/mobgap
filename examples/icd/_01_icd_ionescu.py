"""
ICD Ionescu
========.

This example shows how to use the ICD Ionescu algorithm and how its results compare to the original
matlab implementation.

"""


# %% Helper function to load reference data from the INDIP
def load_reference(datapoint):
    return (
        pd.DataFrame.from_records([{"ICs": wb["InitialContact_Event"]} for wb in datapoint.reference_parameters_["wb"]])
        * datapoint.sampling_rate_hz
    )


def load_matlab_output(datapoint):
    p = datapoint.group_label
    with (
        PACKAGE_ROOT.parent
        / f"example_data/original_results/icd_ionescu/lab/{p.cohort}/{p.participant_id}/ICD_Output.json"
    ).open() as f:
        original_results = json.load(f)[p.time_measure][p.test][p.trial]["SU"]["LowerBack"]["ICs"]

    if not isinstance(original_results, list):
        original_results = [original_results]
    return pd.DataFrame.from_records(original_results).rename({"IC": "ics", "Start": "start", "End": "end"}, axis=1)[
        ["ics"]
    ]


# %% Import useful modules and packages
# Import standard modules
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gaitlink import PACKAGE_ROOT

# Import gaitlink modules
from gaitlink.data import LabExampleDataset
from gaitlink.icd import IcdIonescu

# %%
# Loading some example data
# -------------------------
# .. note :: More infos about data loading can be found in the :ref:`data loading example <data_loading_example>`.
#
# We load example data from the lab dataset together with the INDIP reference system.
# We will use the INDIP "InitialContact_Event" output as ground truth.

lab_example_data = LabExampleDataset(reference_system="INDIP")  # alternatively: "StereoPhoto"

# %%
# Performance on a single lab trial
# ---------------------------------
# Below we apply the algorithm to a lab trail, where we only expect a single gait sequence.
cohort_ = "MS"
participant_id_ = "001"
test_ = "Test5"
trial_ = "Trial1"
short_trial = lab_example_data.get_subset(cohort=cohort_, participant_id=participant_id_, test=test_, trial=trial_)
short_trial_matlab_output = load_matlab_output(short_trial)
short_trial_reference_parameters = load_reference(short_trial)

fs = short_trial.sampling_rate_hz  # sampling rate (Hz)
# if the first sample is t = 0 s --> we don't need to subtract -1 to the gait sequenced
# TODO: Update to new format
# TODO: Check start and end of gait sequence -1 required or not?
s = round(short_trial.reference_parameters_["lwb"][0]["Start"] * fs) - 1  # start of gait sequence (samples)
e = round(short_trial.reference_parameters_["lwb"][0]["End"] * fs) - 1  # end of gait sequence (samples)
gs = short_trial.data["LowerBack"][s : e + 1]  # imu data during gait sequence
short_trial_output = IcdIonescu().detect(gs, sampling_rate_hz=fs)  # output of ICDs
ICs_from_start = (
    short_trial.reference_parameters_["lwb"][0]["Start"] + short_trial_output.icd_list_
)  # ICs (s) from the start of the Trial
print("Reference Parameters:\n\n", short_trial_reference_parameters)
print("\nMatlab Output:\n\n", short_trial_matlab_output)
print("\nPython Output:\n\n", short_trial_output.icd_list_)

# %% Plot and comparison with Matlab
IC_matlab_seconds = [x for x in short_trial_matlab_output["ics"][0]]
IC_matlab_samples = [round(x * fs) for x in short_trial_matlab_output["ics"][0]]
IC_INDIP_seconds = [x / fs for x in short_trial_reference_parameters["ICs"][0]]
IC_INDIP_samples = [round(x) for x in short_trial_reference_parameters["ICs"][0]]

# do two pics--> one general and one more zoomed-in
accV = short_trial.data["LowerBack"]["acc_x"]
plt.close()
fig, ax = plt.subplots()
t = np.arange(1 / fs, (len(accV) + 1) / fs, 1 / fs, dtype=float)
ax.plot(t, accV)
ax.plot(ICs_from_start, accV.array[(ICs_from_start * fs).astype(int)].to_numpy(), "ro", label="Python")
ax.plot(IC_matlab_seconds, accV.array[IC_matlab_samples], "b*", label="Matlab")
ax.plot(IC_INDIP_seconds, accV.array[IC_INDIP_samples], "k+", label="INDIP")
ax.fill_betweenx(np.arange(min(accV) - 1, max(accV) + 1, 0.01), s / fs, e / fs, facecolor="green", alpha=0.2)
plt.xlabel("Time (s)")
plt.ylabel("Vertical Acceleration (m/s^2)")
plt.title("IC detection: HA002 - Test 5 - Trial 2")
plt.legend(loc="upper left")
plt.show()

# %%
# When we plot the output, we can see that both algorithm implementations ignore
# the ICs given by the first and the last element of the gait sequence
