r"""
.. _example_wtd_megaritis_signal:

Wear-time detection with WtdMegaritisSignal
===========================================

This example runs the signal-based wear-time detector on a recording from the bundled lab dataset and plots the
detected intervals over the acceleration signal. The short lab recording demonstrates the API; daily wear-time
estimates require day-long recordings. The detector's default parameters were tuned on one dataset and may need
adjustment for other sensor systems or applications.
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
