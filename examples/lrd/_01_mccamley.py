"""
McCamley L/R detector
=====================

The McCamley L/R detector is a simple algorithm to detect the laterality of initial contacts based on the sign
of the angular velocity signal.
We use a modified version of the original McCamley algorithm, which includes a smoothing filter to reduce the
influence of noise on the detection.

This example shows how to use the algorithm and compares the output to the reference labels on some example data.
"""

from gaitlink.data import LabExampleDataset

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
single_test = example_data.get_subset(
    cohort="MS", participant_id="001", test="Test11", trial="Trial1"
)

imu_data = single_test.data["LowerBack"]
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
#
# We will use the `GsIterator` to iterate over the gait sequences and apply the algorithm to each wb.
# Note, that we use the ``ic_list`` result key, as the output of all L/R detectors is identical to the output of the
# IC-detectors, but with an additional ``lr`` column.
from gaitlink.pipeline import GsIterator
from gaitlink.lrd import LrdMcCamley

iterator = GsIterator()

for (gs, data), result in iterator.iterate(imu_data, reference_wbs):
    result.ic_list = (
        LrdMcCamley()
        .detect(data, ic_list=ref_ics_rel_to_gs.loc[gs.id], sampling_rate_hz=sampling_rate_hz)
        .ic_lr_list_
    )

detected_ics = iterator.results_.ic_list
detected_ics
