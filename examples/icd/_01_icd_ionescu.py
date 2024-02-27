"""
ICD Ionescu
===========

This example shows how to use the icd Ionescu algorithm and how its results compare to the original
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
reference_wbs = single_test.reference_parameters_.wb_list

sampling_rate_hz = single_test.sampling_rate_hz
ref_ics = single_test.reference_parameters_.ic_list

reference_wbs

# %%
# Applying the algorithm
# ----------------------
# Below we apply the shin algorithm to a lab trial.
# We will use the `GsIterator` to iterate over the gait sequences and apply the algorithm to each wb.
from gaitlink.pipeline import GsIterator

iterator = GsIterator()

for (gs, data), result in iterator.iterate(imu_data, reference_wbs):
    result.ic_list = IcdIonescu().detect(data, sampling_rate_hz=sampling_rate_hz).ic_list_

detected_ics = iterator.results_.ic_list
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
    for i, gs in enumerate(original_results, start=1):
        ics[i] = pd.DataFrame({"ic": gs["IC"]}).rename_axis(index="ic_id")

    return (pd.concat(ics, names=["wb_id", ics[1].index.name]) * datapoint.sampling_rate_hz).astype(int)


detected_ics_matlab = load_matlab_output(single_test)
detected_ics_matlab
# %%
# Plotting the results
# --------------------
# With that we can compare the python, matlab and ground truth results.
# We zoom in into one of the gait sequences to better see the output.
# We can make a couple of main observations:
#
# 1. The python version finds the same ICs as the matlab version, but wil a small shift to the left (around 2-5
#    samples/20-50 ms).
#    This is likely due to some differences in the downsampling process.
# 2. Compared to the ground truth reference, both versions detect the IC too early most of the time.
# 3. Both algorithms can not detect the first IC of the gait sequence.
#    However, this is expected, as per definition, this first IC marks the start of the WB in the reference system.
#    Hence, there are no samples before that point the algorithm can use to detect the IC.
imu_data.reset_index(drop=True).plot(y="acc_x")

plt.plot(ref_ics["ic"], imu_data["acc_x"].iloc[ref_ics["ic"]], "o", label="ref")
plt.plot(detected_ics["ic"], imu_data["acc_x"].iloc[detected_ics["ic"]], "x", label="icd_ionescu_py")
plt.plot(detected_ics_matlab["ic"], imu_data["acc_x"].iloc[detected_ics_matlab["ic"]], "+", label="icd_ionescu_matlab")
plt.xlim(reference_wbs.iloc[2]["start"] - 50, reference_wbs.iloc[2]["end"] + 50)
plt.legend()
plt.show()

# %%
# Evaluation of the algorithm against a reference
# --------------------------------------------------
# Let's quantify how the Python output compares to the reference labels.
# To do this, we use the :func:`~gaitlink.icd.evaluation.evaluate_initial_contact_list` function to compare the detected
# ICs to the ground truth ICs. With the `tolerance` parameter, we can specify the maximum allowed deviation in samples.
# Consequently, the tolerance parameter should be chosen with respect to the sampling rate of the data.
# In this case, it is set to 20 samples, which corresponds to 200 ms.
from gaitlink.icd.evaluation import evaluate_initial_contact_list

matches_all_wb = []
# We iterate over all walking bouts and perform the evaluation for each
for wb_id, wb_detected in detected_ics.groupby("wb_id"):
    wb_ref = ref_ics[ref_ics.index.get_level_values("wb_id") == wb_id][["ic"]]
    matches = evaluate_initial_contact_list(wb_detected, wb_ref, tolerance=20)
    # add wb_id to matches
    matches["wb_id"] = wb_id
    matches_all_wb.append(matches)
matches = pd.concat(matches_all_wb).set_index("wb_id")
# %%
# The function returns a DataFrame containing the ID of the detected ICs (`"ic_id_detected"`) and the reference ICs
# (`"ic_id_reference"`) that they match to. If there is no match for the respective IC, the IC ID is set to `NaN`.
# The column `"match_type"` indicates the type of every match, i.e. `tp` for true positive, `fp` for false
# positive, and `fn` for false negative.

print("Matched Initial Contacts:\n\n", matches)

# %%
# Based on the tp, fp, and fn contacts, common performance metrics such as F1 score, precision,
# and recall can be calculated.
# For this purpose, the :func:`~gaitlink.utils.evaluation.precision_recall_f1_score` function can be used.
# It returns a dictionary containing the metrics for the specified categorized intervals DataFrame.
from gaitlink.utils.evaluation import precision_recall_f1_score

prec_rec_f1_dict = precision_recall_f1_score(matches)

print("Performance Metrics:\n\n", prec_rec_f1_dict)

# %%
# To calculate the whole range of possible performance metrics in one go,
# we can use the :func:`~gaitlink.icd.evaluation.calculate_icd_performance_metrics` function.
# It returns a dictionary containing all metrics for the specified detected and reference initial contact lists.
from gaitlink.icd.evaluation import calculate_icd_performance_metrics

metrics_all = calculate_icd_performance_metrics(matches=matches)

print("Performance Metrics:\n\n", metrics_all)
