"""
McCamley L/R Classifier
=======================

The McCamley L/R classifier is a simple algorithm to detect the laterality of initial contacts based on the sign
of the angular velocity signal.
We use a modified version of the original McCamley algorithm, which includes a smoothing filter to reduce the
influence of noise on the detection.

This example shows how to use the algorithm and compares the output to the reference labels on some example data.
"""

import pandas as pd

from mobgap.data import LabExampleDataset

# %%
# Loading some example data
# -------------------------
# .. note :: More infos about data loading can be found in the :ref:`data loading example <data_loading_example>`.
#
# We load example data from the lab dataset together with the INDIP reference system.
# We will use the INDIP "InitialContact_Event" output as ground truth.
#
# We only use the data from the "simulated daily living" activity test from a single particomand.

example_data = LabExampleDataset(reference_system="INDIP", reference_para_level="wb")
single_test = example_data.get_subset(cohort="MS", participant_id="001", test="Test11", trial="Trial1")

imu_data = single_test.data_ss
reference_wbs = single_test.reference_parameters_.wb_list

sampling_rate_hz = single_test.sampling_rate_hz
ref_ics = single_test.reference_parameters_.ic_list
ref_ics_rel_to_gs = single_test.reference_parameters_relative_to_wb_.ic_list

reference_wbs
# %%
# Applying the algorithm using reference ICs
# ------------------------------------------
# We use the McCamley algorithm to detect the laterality of the initial contacts.
# For this we need the IMU data and the indices of the initial contacts per GS.
# To focus this example on the L/R detection, we use the reference ICs from the INDIP system as input.
# In a real application, we would use the output of the IC-detectors as input.
#
# We will use the `GsIterator` to iterate over the gait sequences and apply the algorithm to each wb.
# Note, that we use the ``ic_list`` result key, as the output of all L/R detectors is identical to the output of the
# IC-detectors, but with an additional ``lr_label`` column.
from mobgap.lrc import LrdMcCamley
from mobgap.pipeline import GsIterator

iterator = GsIterator()

for (gs, data), result in iterator.iterate(imu_data, reference_wbs):
    result.ic_list = (
        LrdMcCamley().predict(data, ic_list=ref_ics_rel_to_gs.loc[gs.id], sampling_rate_hz=sampling_rate_hz).ic_lr_list_
    )

detected_ics = iterator.results_.ic_list
detected_ics

# %%
# Compare the results to the reference
# ------------------------------------
# We compare the detected initial contacts to the reference labels.
# One easy way to compare the results is to visualize them as colorful bars.

import matplotlib.pyplot as plt


def plot_lr(ref, detected):
    fig, ax = plt.subplots(figsize=(15, 5))
    # We plot one box either (red or blue depending on the laterality) for each detected IC ignoring the actual time
    for (_, row), (_, ref_row) in zip(detected.iterrows(), ref.iterrows()):
        ax.plot([row["ic"], row["ic"]], [0, 0.98], color="r" if row["lr_label"] == "left" else "b", linewidth=5)
        ax.plot(
            [ref_row["ic"], ref_row["ic"]], [1.02, 2], color="r" if ref_row["lr_label"] == "left" else "b", linewidth=5
        )

    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Detected", "Reference"])
    return fig, ax


fig, _ = plot_lr(ref_ics, detected_ics)
fig.show()

# %%
# If we zoom in on a longer WB, we can see that for some ICs the L/R label does not match.
# But, in particular for regular gait in the center of the WB, the labels match quite well.

fig, ax = plot_lr(ref_ics, detected_ics)
ax.set_xlim(12000, 15000)
fig.show()

# %%
# We can also quantify the agreement between the detected and the reference labels using typical classification metrics.
from sklearn.metrics import classification_report

pd.DataFrame(
    classification_report(ref_ics.lr_label, detected_ics.lr_label, target_names=["left", "right"], output_dict=True)
).T
