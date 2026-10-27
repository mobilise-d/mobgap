"""
Secondary outcomes after the primary pipeline
=============================================

Calculate secondary outcomes after detecting walking bouts and turns. The primary pipeline supplies accepted WBs,
retained strides, detected turns and the recorded orientation corrections. Two explicit loops prepare the WB and turn
signals, while :class:`~mobgap.pipeline.SdmoIterator` assembles one outcome row per window.

The iterator returns the existing ``Region`` type. Extra metadata, including the method-specific correction value,
stays in the source DataFrame and is read by the region's identifier inside the loop.
"""

# %%
# Run the primary pipeline
# ------------------------
# This trial includes several walking bouts and turns. Disable secondary calculation and aggregation inside the
# primary pipeline so that we can calculate them afterwards. Keep intermediate results to access the recorded
# per-GS corrections. With per-GS reorientation, gait detection must support sensor-frame inputs.
from mobgap.data import LabExampleDataset
from mobgap.gait_sequences import GsdIluzAdaptiveGravity
from mobgap.pipeline import MobilisedPipelineHealthy, SdmoIterator
from mobgap.re_orientation import (
    ReorientationMethodDM,
    get_orientation_for_sequences,
)

short_trial = LabExampleDataset(reference_system="INDIP").get_subset(
    cohort="HA", participant_id="001", test="Test11", trial="Trial1"
)
sampling_rate_hz = short_trial.sampling_rate_hz
sensor_data = short_trial.data_ss
pipeline = MobilisedPipelineHealthy(
    gait_sequence_detection=GsdIluzAdaptiveGravity(),
    per_gs_reorientation=ReorientationMethodDM(),
    retain_intermediate_results=True,
    sdmo_calculation=None,
    sdmo_aggregation=None,
).safe_run(short_trial)

# %%
# Prepare corrections once, outside the loops
# -------------------------------------------
# The per-GS iterator's public raw results retain the correction belonging to each detected GS. Sub-iterations are
# excluded here. Correction values remain specific to ReorientationMethodDM, which returns two scipy rotations.
# Do not re-estimate orientation independently for each WB or turn.
import pandas as pd
from scipy.spatial.transform import Rotation

gs_corrections = {
    entry.input.region.id: entry.result.reorientation_result
    for entry in pipeline.gs_iterator_.raw_results_
    if entry.iteration_name == "__main__"
}
reorientation_sections = pipeline.gs_list_.assign(
    orientation=lambda rows: pd.Series(
        {
            gs_id: correction.correction_rotations
            for gs_id, correction in gs_corrections.items()
        },
        dtype=object,
    ).reindex(rows.index),
    correction_applied=lambda rows: pd.Series(
        {
            gs_id: correction.correction_applied
            for gs_id, correction in gs_corrections.items()
        }
    ).reindex(rows.index),
)
identity_correction = (Rotation.identity(), Rotation.identity())
wbs = pipeline.per_wb_parameters_.assign(
    orientation=lambda rows: get_orientation_for_sequences(
        rows, reorientation_sections, default=identity_correction
    )
)
turns = pipeline.raw_turn_list_.assign(
    orientation=lambda rows: get_orientation_for_sequences(
        rows, reorientation_sections, default=identity_correction
    )
)

# %%
# .. warning::
#    Signal outcomes require correctly oriented body-frame data. Nominal column conversion alone cannot correct a
#    misoriented sensor. A WB need not be fully contained in a GS: the helper raises if it only partly overlaps any
#    section requiring correction. It does not extend that correction into gaps or across other sections. Selecting
#    or replaying a recorded fallback also does not establish that orientation was resolved; inspect the correction
#    metadata when this matters for your analysis.
#
# On this aligned example recording, the recorded corrections are nominal. With reorientation disabled, callers who
# know their sensor is correctly aligned can use ``to_body_frame(window)`` directly instead.

# %%
# Walking-bout outcomes
# ---------------------
# Select the non-turn calculators from the composition. The loop owns signal preparation and stride-coordinate
# rebasing. Retain the full accepted WB signal, including turns and gaps between retained strides.
from mobgap.signal_based import MobilisedSDMO, TurnSDMO

wb_calculator = MobilisedSDMO(
    calculators=tuple(
        (name, calculator)
        for name, calculator in MobilisedSDMO.PredefinedParameters.default[
            "calculators"
        ]
        if calculator.result_level == "wb"
    )
)
wb_iterator = SdmoIterator()
for (wb, window), result in wb_iterator.iterate(sensor_data, wbs):
    correction = wbs.loc[wb.id, "orientation"]
    prepared = pipeline.per_gs_reorientation.apply_correction(
        window, correction
    )
    strides = pipeline.per_stride_parameters_.loc[wb.id].copy()
    strides.loc[:, ["start", "end"]] -= wb.start
    result.parameters = (
        wb_calculator.clone()
        .calculate(
            prepared, sampling_rate_hz=sampling_rate_hz, stride_list=strides
        )
        .signal_based_parameters_
    )

per_wb_sdmos = wb_iterator.results_.parameters
per_wb_sdmos

# %%
# Individual turn outcomes
# ------------------------
# Every detected turn is processed, including turns outside accepted WBs. Its full identity remains
# ``(gs_id, turn_id)``; we do not infer WB membership from the GS identifier.
# TurnSDMO currently accepts a turn table, so supply the one turn spanning the prepared local window. Only its
# outcome columns are assembled. Recording-local start/end and other metadata are joined back from the source.
turn_calculator = TurnSDMO()
turn_iterator = SdmoIterator()
for (turn, window), result in turn_iterator.iterate(sensor_data, turns):
    correction = turns.loc[turn.id, "orientation"]
    prepared = pipeline.per_gs_reorientation.apply_correction(
        window, correction
    )
    local_turn = turns.loc[[turn.id]].drop(columns="orientation").copy()
    local_turn.loc[:, "start"] = 0
    local_turn.loc[:, "end"] = len(window)
    result.parameters = (
        turn_calculator.clone()
        .calculate(
            prepared, sampling_rate_hz=sampling_rate_hz, turn_list=local_turn
        )
        .per_turn_parameters_.drop(columns=local_turn.columns)
    )

per_turn_sdmos = turns.drop(columns="orientation").join(
    turn_iterator.results_.parameters
)
per_turn_sdmos

# %%
# Optional aggregation across WBs
# -------------------------------
# Iterator assembly and statistical aggregation are separate operations. Supply only outcome columns and required
# duration/grouping metadata to SDMOAggregator; the orientation payload is not an outcome. Each WB has equal weight.
# Primary-DMO validity thresholds are not automatically applied to secondary outcomes. Turn rows remain separate.
from mobgap.aggregation import SDMOAggregator

aggregated_sdmos = (
    SDMOAggregator()
    .aggregate(per_wb_sdmos.join(wbs[["duration_s"]]))
    .aggregated_data_
)
aggregated_sdmos
