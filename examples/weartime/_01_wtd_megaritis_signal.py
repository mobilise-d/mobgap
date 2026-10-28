r"""
.. _example_wtd_megaritis_signal:

Wear-time detection with WtdMegaritisSignal
===========================================

This example runs the signal-based wear-time detector on a recording from the bundled lab dataset and plots the
detected intervals over the acceleration signal. The short lab recording demonstrates the API; daily wear-time
filtering for clinical aggregation requires recordings covering the intended calendar day. The final section shows
how to use the waking-hours estimate to determine day eligibility. The detector's default parameters were tuned on
one dataset and may need adjustment for other sensor systems or applications.
"""

import matplotlib.pyplot as plt
from mobgap.data import LabExampleDataset
from mobgap.weartime import WtdMegaritisSignal
from mobgap.weartime.pipeline import WtdEmulationPipeline

# %%
# Load one lab trial. The pipeline reads its sensor data, converts it to the body frame, and passes its metadata to the
# detector.
recording = LabExampleDataset(
    reference_system="INDIP", reference_para_level="wb"
).get_subset(cohort="MS", participant_id="001", test="Test11", trial="Trial1")
pipeline = WtdEmulationPipeline(WtdMegaritisSignal()).run(recording)

# %%
# Detected intervals use sample indices relative to the recording start. ``end`` is exclusive.
pipeline.weartime_list_

# %%
# Total wear time is available in minutes. The detector also reports its runtime through ``perf_``.
pipeline.total_weartime_min_

# %%
pipeline.algo_.perf_

# %%
# Plot detected wear intervals over the body-frame vertical acceleration. This lab trial was worn throughout, so a
# detected interval covering most of the recording is expected.
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(pipeline.algo_.data["acc_is"].to_numpy(), label="acc_is", alpha=0.7)
for i, (start, end) in enumerate(
    pipeline.weartime_list_[["start", "end"]].itertuples(index=False)
):
    ax.axvspan(
        start,
        end,
        alpha=0.2,
        color="tab:orange",
        label="detected wear" if i == 0 else None,
    )
ax.set_xlabel("Sample")
ax.set_ylabel("Acceleration (m/s²)")
ax.legend()
plt.tight_layout()
plt.show()

# %%
# Daily wear time for CVS aggregation
# -----------------------------------
# The default Mobilise-D gait pipelines do not run wear-time detection automatically. Run ``WtdEmulationPipeline``
# alongside the gait pipeline on each daily recording, then use its waking-hours wear time to filter daily DMO rows.
#
# The following snippet is for a user-supplied ``daily_recording``, a single dataset datapoint whose sensor data has a
# timezone-aware ``DatetimeIndex`` in the participant's local timezone and lies within one calendar date. It is not
# executed on the short lab trial above. Split multi-day recordings into daily datapoints first; the waking-hours
# summary rejects recordings that cross local midnight.
#
# .. code-block:: python
#
#    from datetime import time
#
#    daily_wear = WtdEmulationPipeline(
#        WtdMegaritisSignal(waking_hours=(time(7), time(22)))
#    ).run(daily_recording)
#    waking_wear_hours = daily_wear.total_weartime_during_waking_min_ / 60
#    is_valid_day = waking_wear_hours > 12
#
# The CVS day criterion is strictly more than 12 hours of wear during 07:00 to 22:00 local time. Use ``is_valid_day``
# to retain the daily DMO row for the same participant, visit, and date before weekly aggregation. Keep the day-level
# gait outcomes based on the full calendar day; the waking-hours window only determines wear-time eligibility.
#
# Wear time counts all periods when the device is worn, including periods without walking. It must not be replaced
# by walking duration or by ``total_weartime_min_``, which also includes wear outside the waking-hours window.
# Partial-day recordings contribute only the detected wear within their observed portion of the window; check
# recording coverage separately.
#
# Finally, average daily outcomes over valid days and retain weekly results with at least three valid days in the
# intended seven-day assessment. See :doc:`/guides/clinical_research_outputs` for the complete aggregation procedure.
