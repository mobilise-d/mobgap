"""
Build a full pipeline with all the steps
========================================

This example shows how to build a full gait analysis pipeline using the mobgap package.
Note, that we provide pre-built pipelines for common use-cases in the package.
Checkout the examples for those, if you want to understand how to use them.

"""

import pandas as pd

# TODO: This example is WIP and will be expanded once the respective algorithms are implemented.
# %%
# Load example data
# -----------------
# We load example data from the lab dataset, and we will use a single long-trail from a "MS" participant for this
# example.
from mobgap.data import LabExampleDataset
from mobgap.utils.interpolation import naive_sec_paras_to_regions

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
from mobgap.lrc import strides_list_from_ic_lr_list

stride_list = results.ic_list.groupby("gs_id", group_keys=False).apply(
    strides_list_from_ic_lr_list
)
stride_list
# %%
# This initial stride list is completely unfiltered, and might contain very long strides, in areas where initial
# contacts were not detected, or the participant was not walking for a short moment.
# The stride list will be filtered later as part of the WB assembly.
from mobgap.utils.array_handling import create_multi_groupby

stride_list_with_paras = create_multi_groupby(
    stride_list,
    pd.concat([results.cad_per_sec.reset_index("r_gs_id", drop=True)]),
    "gs_id",
).apply(naive_sec_paras_to_regions)

stride_list_with_paras
# %%
# Now the final strides are regrouped into walking bouts.
# For this we ignore which gait sequence the strides belong to, hence we remove the ``gs_id`` from the index, but keep
# it around as column for debugging.
# However, to keep a unique stride identifier, we combine the ``gs_id`` and the stride index.
from mobgap.wba import IntervalDurationCriteria, StrideSelection, WbAssembly

stride_list_with_paras = stride_list_with_paras.reset_index("gs_id").rename(columns={"gs_id": "original_gs_id"})

ss = StrideSelection(
    rules=[
        (
            "stride_duration_thres",
            IntervalDurationCriteria(min_duration_s=0.2, max_duration_s=3.0),
        )
    ]
).filter(stride_list_with_paras, sampling_rate_hz=sampling_rate_hz)
wba = WbAssembly().assemble(
    ss.filtered_stride_list_, sampling_rate_hz=sampling_rate_hz
)

final_strides = wba.annotated_stride_list_
final_strides
# %%
# We also have meta information about the WBs available.
per_wb_params = wba.wb_meta_parameters_
per_wb_params

# %%
# We extend them further with the per-stride parameters.
params_to_aggregate = ["cad_spm", "duration_s", "stride_length_m"]
per_wb_params = pd.concat([per_wb_params, final_strides.reindex(columns=params_to_aggregate).groupby(["wb_id"]).mean()], axis=1)

per_wb_params

# %%
# For each WB we can then apply thresholds to check if the calculated parameters are within the expected range.
from mobgap.aggregation import apply_thresholds, get_mobilised_dmo_thresholds

thresholds = get_mobilised_dmo_thresholds()

per_wb_params_mask = apply_thresholds(
    per_wb_params,
    thresholds,
    cohort=long_trial.participant_metadata["cohort"],
    height_m=long_trial.participant_metadata["Height"],
    measurement_condition="laboratory",
)
