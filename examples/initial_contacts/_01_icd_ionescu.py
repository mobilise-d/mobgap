r"""
.. _icd_ionescu:

ICD Ionescu
===========

This example shows how to use the initial_contacts Ionescu algorithm and how its results compare to the original
matlab implementation.

"""

# %%
# Import useful modules and packages
import matplotlib.pyplot as plt
import pandas as pd
from mobgap.data import LabExampleDataset
from mobgap.initial_contacts import IcdIonescu

# %%
# Loading some example data
# -------------------------
# .. note :: More infos about data loading can be found in the :ref:`data loading example <data_loading_example>`.
#
# We load example data from the lab dataset together with the INDIP reference system.
# We will use the INDIP "InitialContact_Event" output as ground truth.

example_data = LabExampleDataset(
    reference_system="INDIP", reference_para_level="wb"
)

# %%
# Performance on a single lab trial
# ---------------------------------
# Below we apply the algorithm to a lab trail, where we only expect a single gait sequence.
#
# Like most algorithms, the algorithm requires the data to be in body frame coordinates.
# As we know the sensor was well aligned, we can just use ``to_body_frame`` to transform the data.
from mobgap.utils.conversions import to_body_frame

single_test = example_data.get_subset(
    cohort="HA", participant_id="001", test="Test11", trial="Trial1"
)
imu_data = to_body_frame(single_test.data_ss)
reference_wbs = single_test.reference_parameters_.wb_list

sampling_rate_hz = single_test.sampling_rate_hz
ref_ics = single_test.reference_parameters_.ic_list

reference_wbs

# %%
# Applying the algorithm
# ----------------------
# Below we apply the shin algorithm to a lab trial.
# We will use the `GsIterator` to iterate over the gait sequences and apply the algorithm to each wb.
from mobgap.pipeline import GsIterator

iterator = GsIterator()

for (gs, data), result in iterator.iterate(imu_data, reference_wbs):
    result.ic_list = (
        IcdIonescu().detect(data, sampling_rate_hz=sampling_rate_hz).ic_list_
    )

detected_ics = iterator.results_.ic_list
detected_ics

# %%
# Matlab Outputs
# --------------
# To check if the algorithm was implemented correctly, we compare the results to the matlab implementation.
import json

from mobgap import PROJECT_ROOT


def load_matlab_output(datapoint):
    p = datapoint.group_label
    with (
        PROJECT_ROOT
        / f"example_data/original_results/icd_ionescu/lab/{p.cohort}/{p.participant_id}/SD_Output.json"
    ).open() as f:
        original_results = json.load(f)["SD_Output"][p.time_measure][p.test][
            p.trial
        ]["SU"]["LowerBack"]["SD"]

    if not isinstance(original_results, list):
        original_results = [original_results]

    ics = {}
    for i, gs in enumerate(original_results, start=1):
        ics[i] = pd.DataFrame({"ic": gs["IC"]}).rename_axis(index="step_id")

    return (
        pd.concat(ics, names=["wb_id", ics[1].index.name])
        * datapoint.sampling_rate_hz
    ).astype("int64")


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
imu_data.reset_index(drop=True).plot(y="acc_is")

plt.plot(
    ref_ics["ic"], imu_data["acc_is"].iloc[ref_ics["ic"]], "o", label="ref"
)
plt.plot(
    detected_ics["ic"],
    imu_data["acc_is"].iloc[detected_ics["ic"]],
    "x",
    label="icd_ionescu_py",
)
plt.plot(
    detected_ics_matlab["ic"],
    imu_data["acc_is"].iloc[detected_ics_matlab["ic"]],
    "+",
    label="icd_ionescu_matlab",
)
plt.xlim(reference_wbs.iloc[2]["start"] - 50, reference_wbs.iloc[2]["end"] + 50)
plt.legend()
plt.show()

# %%
# Evaluation of the algorithm against a reference
# -----------------------------------------------
# To quantify how the Python output compares to the reference labels, we are providing a range of evaluation functions.
# See the :ref:`example on ICD evaluation <icd_evaluation>` for more details.
