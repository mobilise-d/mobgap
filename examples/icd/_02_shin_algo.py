"""
Shin Algo
=========

This example shows how to use the improved Shin algorithm and some examples on how the results compare to the original
matlab implementation.

"""

import pandas as pd
from matplotlib import pyplot as plt

from gaitlink.data import LabExampleDataset
from gaitlink.icd import IcdShinImproved

# %%
# Loading data
# ------------
# .. note :: More infos about data loading can be found in the :ref:`data loading example <data_loading_example>`.
# We load example data from the lab dataset together with the INDIP reference system.
# We will use the INDIP output for initial contacts ("ic") as ground truth.

example_data = LabExampleDataset(reference_system="INDIP", reference_para_level="wb")

single_test = example_data.get_subset(cohort="HA", participant_id="001", test="Test11", trial="Trial1")
imu_data = single_test.data["LowerBack"]

# importing the start and end of each walking bout identified from the INDIP system (ground truth)
reference_wbs = single_test.reference_parameters_.walking_bouts

sampling_rate_hz = single_test.sampling_rate_hz
ref_ics = single_test.reference_parameters_.initial_contacts

# %%
# Applying the algorithm
# ----------------------
# Below we apply the shin algorithm to a lab trial.
# We will use the `GsIterator` to iterate over the gait sequences and apply the algorithm to each wb.
from gaitlink.pipeline import GsIterator

iterator = GsIterator()

for (gs, data), result in iterator.iterate(imu_data, reference_wbs):
    result.initial_contacts = IcdShinImproved(axis="norm").detect(data, sampling_rate_hz=sampling_rate_hz).ic_list_

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
        / f"example_data/original_results/icd_shin_improved/lab/{p.cohort}/{p.participant_id}/SD_Output.json"
    ).open() as f:
        original_results = json.load(f)["SD_Output"][p.time_measure][p.test][p.trial]["SU"]["LowerBack"]["SD"]

    if not isinstance(original_results, list):
        original_results = [original_results]

    ics = {}
    for i, gs in enumerate(original_results):
        ics[i] = pd.DataFrame({"ic": gs["IC"]})

    return (pd.concat(ics) * datapoint.sampling_rate_hz).astype(int)


detected_ics_matlab = load_matlab_output(single_test)
detected_ics_matlab
# %%
# Plotting the results
# --------------------
# With that we can compare the python, matlab and ground truth results.

# Updated the plot to show the ICs for each walking bout in files with multiple bouts
imu_data.reset_index(drop=True).plot(y="acc_x")

plt.plot(ref_ics["ic"], imu_data["acc_x"].iloc[ref_ics["ic"]], "o", label="ref")
plt.plot(detected_ics["ic"], imu_data["acc_x"].iloc[detected_ics["ic"]], "x", label="shin_algo_py")
plt.plot(detected_ics_matlab["ic"], imu_data["acc_x"].iloc[detected_ics_matlab["ic"]], "+", label="shin_algo_matlab")
plt.xlim(3800, 4750)
plt.legend()
plt.show()

# %%
# We can see that the python version finds one IC for each IC of the matlab version.
# However, it is delayed by a couple of samples.
# The full reason of that is not clear, but it is likely due to the differences in indexing and small differences in
# how we perform down and up sampling.
