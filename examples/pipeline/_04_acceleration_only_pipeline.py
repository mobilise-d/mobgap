"""
Experimental acceleration-only pipeline
=======================================

.. warning:: This acceleration-only pipeline has not been properly validated. Its results must not be treated as
             validated Mobilise-D outcomes.

.. warning:: This is not the approved Mobilise-D pipeline. The approved pipeline always requires gyroscope data.

This example runs a custom pipeline on lower-back acceleration data using LrcBenMansour instead of the default
left/right detector. It also disables turn detection.

In this configuration, the pipeline uses only acceleration data and could run with a sensor that has no gyroscope,
potentially reducing cost and battery use. Without turn detection and gyroscope data, the pipeline can calculate the main
outcome metrics, but most secondary outcomes are unavailable. It therefore cannot produce the full set of Mobilise-D
outputs.
"""

# %%
# Use one lab recording and retain only its acceleration channels.
from mobgap.consts import SF_ACC_COLS
from mobgap.data import GaitDatasetFromData, LabExampleDataset
from mobgap.laterality import LrcBenMansour
from mobgap.pipeline import GenericMobilisedPipeline

recording = LabExampleDataset().get_subset(
    cohort="HA", participant_id="001", test="Test5", trial="Trial2"
)
acc_only_recording = GaitDatasetFromData(
    {"test": {"LowerBack": recording.data_ss[SF_ACC_COLS]}},
    recording.sampling_rate_hz,
    _participant_metadata={"test": recording.participant_metadata},
    _recording_metadata={"test": recording.recording_metadata},
)[0]

# %%
# The input contains only acceleration channels.
acc_only_recording.data_ss.head()

# %%
# The regular-walking defaults use acceleration for gait sequences, initial contacts, cadence, stride length, and
# walking speed. Replace gyroscope-based laterality classification and disable gyroscope-based turn detection.
acc_only_pipeline = GenericMobilisedPipeline(
    **(
        GenericMobilisedPipeline.PredefinedParameters.regular_walking
        | {"laterality_classification": LrcBenMansour(), "turn_detection": None}
    )
)
acc_only_pipeline.run(acc_only_recording)

# %%
# One row per detected walking bout. The four columns below are the main walking-bout outcomes.
acc_only_pipeline.per_wb_parameters_[
    ["duration_s", "cadence_spm", "stride_length_m", "walking_speed_mps"]
]
