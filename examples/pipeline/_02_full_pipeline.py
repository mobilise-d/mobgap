"""
Build a full pipeline with all the steps
========================================

This example shows how to build a full gait analysis pipeline using the mobgap package.
Note, that we provide pre-built pipelines for common use-cases in the package.
Checkout the examples for those, if you want to understand how to use them.

"""
# TODO: This example is WIP and will be expanded once the respective algorithms are implemented.

# %%
# Load example data
# -----------------
# We load example data from the lab dataset, and we will use a single long-trail from a "MS" participant for this
# example.
from mobgap.data import LabExampleDataset

lab_example_data = LabExampleDataset(reference_system="INDIP")
long_trial = lab_example_data.get_subset(
    cohort="MS", participant_id="001", test="Test11", trial="Trial1"
)
imu_data = long_trial.data_ss
sampling_rate_hz = long_trial.sampling_rate_hz

# %%
# Step 1: Gait Sequence Detection
# -------------------------------
from mobgap.gsd import GsdIluz

gsd = GsdIluz()
gsd.detect(imu_data, sampling_rate_hz=sampling_rate_hz)

gait_sequences = gsd.gs_list_
gait_sequences

# %%
# Starting from here, all the processing will happen per gait sequence.
# We will go through the steps just for a single gait sequence first and later put everything in a loop.
first_gait_sequence = gait_sequences.iloc[0]
first_gait_sequence_data = imu_data.iloc[
    first_gait_sequence.start : first_gait_sequence.end
]

# %%
# Step 2: Initial Contact Detection
# ---------------------------------

from mobgap.icd import IcdShinImproved

icd = IcdShinImproved()
icd.detect(first_gait_sequence_data, sampling_rate_hz=sampling_rate_hz)
ic_list = icd.ic_list_
ic_list

# %%
# Step 2.5: Laterality Detection
# ------------------------------
# For each IC we want to detect the laterality.
from mobgap.lrc import LrcUllrich

lrc = LrcUllrich()
lrc.predict(
    first_gait_sequence_data, ic_list, sampling_rate_hz=sampling_rate_hz
)
ic_list = lrc.ic_lr_list_

# %%
# Gait Sequence Refinement
# -----------------------------------
# After detecting the ICs within the gait sequence, we can refine the gait sequence using the ICs.
# Basically, we restrict the area of the gait sequence to the area between the first and the last IC.
# This should ensure that the subsequent steps are only getting data that contains detectable gait.
from mobgap.icd import refine_gs

refined_gait_sequence, refined_ic_list = refine_gs(ic_list)
refined_gait_sequence_data = first_gait_sequence_data.iloc[
    refined_gait_sequence.iloc[0].start : refined_gait_sequence.iloc[0].end
]

# %%
# Step 3: Cadence Calculation
# ---------------------------
from mobgap.cad import CadFromIc

cad = CadFromIc()
cad.calculate(
    refined_gait_sequence_data,
    refined_ic_list,
    sampling_rate_hz=sampling_rate_hz,
)

cad_per_sec = cad.cad_per_sec_
cad_per_sec

# %%
# Step 4: Stride Length Calculation
# ---------------------------------
# TODO: Add stride length calculation here.

# %%
# After going through the steps for a single gait sequence, we would then put all the data together to calculate final
# results per WB.
# But let's first put all the processing into an easy-to-read loop.
#
# Actual Pipeline
# ---------------
# We first define all the algorithms we want to use.
from mobgap.cad import CadFromIc
from mobgap.gsd import GsdIluz
from mobgap.icd import IcdShinImproved, refine_gs
from mobgap.lrc import LrcUllrich

gsd = GsdIluz()
icd = IcdShinImproved()
lrc = LrcUllrich()
cad = CadFromIc()

# %%
# Then we calculate the gait sequences as before.
gsd.detect(imu_data, sampling_rate_hz=sampling_rate_hz)
gait_sequences = gsd.gs_list_

# %%
# Then we use the nested iterator to go through all the gait sequences and process them.
# Note, that we use the special ``r`` object to store the results of each step and the ``with_subregion`` method to
# elegantly handle the refined gait sequence.
from mobgap.pipeline import GsIterator

gs_iterator = GsIterator()

for (_, gs_data), r in gs_iterator.iterate(imu_data, gait_sequences):
    icd.detect(gs_data, sampling_rate_hz=sampling_rate_hz)
    lrc.predict(gs_data, icd.ic_list_, sampling_rate_hz=sampling_rate_hz)
    r.ic_list = lrc.ic_lr_list_

    refined_gs, refined_ic_list = refine_gs(r.ic_list)

    with gs_iterator.subregion(refined_gs) as ((_, refined_gs_data), rr):
        cad.calculate(
            refined_gs_data, refined_ic_list, sampling_rate_hz=sampling_rate_hz
        )
        rr.cad_per_sec = cad.cad_per_sec_

# %%
# Now we can access all accumulated and offset-corrected results from the iterator.
results = gs_iterator.results_

results.ic_list

# %%
results.cad_per_sec

# %%
# Using the combined results, we want to define walking bouts.
# As walking bouts in the context of Mobilise-D are defined based on strides, we need to turn the ICs into strides and
# the per-second values into per-stride values by using interpolation.
# TODO: This is not implemented yet.
# stride_list = stride_from_ic_list(results.ic_list)
# stride_list_with_paras = interpolate_per_stride(
#     stride_list, cad_per_sec=results.cad_per_sec, stride_length=results.stride_length
# )
#
#
# from mobgap.wba import StrideSelection, WbAssembly
#
# ss = StrideSelection().filter()
# wba = WbAssembly().assemble()
