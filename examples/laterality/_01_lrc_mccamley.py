"""
McCamley L/R Classifier
=======================

The McCamley L/R classifier is a simple algorithm to detect the laterality of initial contacts based on the sign
of the angular velocity signal.
We use a modified version of the original McCamley algorithm, which includes a smoothing filter to reduce the
influence of noise on the detection.

This example shows how to use the algorithm and compares the output to the reference labels on some example data.
"""

from mobgap.data import LabExampleDataset

# %%
# Loading some example data
# -------------------------
# .. note :: More infos about data loading can be found in the :ref:`data loading example <data_loading_example>`.
#
# We load example data from the lab dataset together with the INDIP reference system.
# We will use the INDIP "InitialContact_Event" output as ground truth.
#
# We only use the data from the "simulated daily living" activity test from a single participant.
#
# Like most algorithms, the algorithm requires the data to be in body frame coordinates.
# As we know the sensor was well aligned, we can just use ``to_body_frame`` to transform the data.
from mobgap.utils.conversions import to_body_frame

example_data = LabExampleDataset(
    reference_system="INDIP", reference_para_level="wb"
)
single_test = example_data.get_subset(
    cohort="MS", participant_id="001", test="Test11", trial="Trial1"
)

imu_data = to_body_frame(single_test.data_ss)
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
from mobgap.laterality import LrcMcCamley
from mobgap.pipeline import GsIterator

iterator = GsIterator()
algo = LrcMcCamley()

for (gs, data), result in iterator.iterate(imu_data, reference_wbs):
    result.ic_list = algo.predict(
        data,
        ic_list=ref_ics_rel_to_gs.loc[gs.id].drop("lr_label", axis=1),
        sampling_rate_hz=sampling_rate_hz,
    ).ic_lr_list_

detected_ics = iterator.results_.ic_list
detected_ics.assign(ref_lr_label=ref_ics.lr_label)

# %%
# We can see that for most ICs we correctly identify the laterality.
# If you want to learn more about evaluating the algorithm output, you can check the
# :ref:`evaluation example <lrc_evaluation>`.
