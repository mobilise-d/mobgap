"""
Signal-based parameters
=======================

This example shows how to use the :class:`.MobilisedSDMO` for calculating signal-based parameters.

First we load some data.

Example Data
------------
We load example data from the lab dataset together with the INDIP reference system.
We will use a single short-trail from the "HA" participant for this example, as it only contains a single gait sequence.
The signal-based parameters are calculated for each walking bout in the Mobilise-D pipelines, however, the calculator
(algorithm) classes are designed to work with any signal window (although note that some parameters require to have
some characteristics such as certain number of strides).

"""

from mobgap.data import LabExampleDataset

lab_example_data = LabExampleDataset(reference_system="INDIP")
short_trial = lab_example_data.get_subset(
    cohort="HA", participant_id="001", test="Test5", trial="Trial2"
)

# %%
# MobilisedSDMO
# ----
# To demonstrate the usage of the pipeline :class:`.MobilisedSDMO`, we use the detected walking bout from the reference
# system as input.
reference_strides = (
    short_trial.reference_parameters_relative_to_wb_.stride_parameters
)
reference_strides

# %%
# We pick the first walking bout for this example.
wb_id = reference_strides.index[0][0]
reference_strides = reference_strides.loc[wb_id]
data_in_wb = short_trial.data["LowerBack"].iloc[
    reference_strides.start.iloc[0] : reference_strides.end.iloc[-1]
]

# %%
# The data is required to be in body frame coordinates.
# As we know the sensor was well aligned, we can just use ``to_body_frame`` to transform the data.
from mobgap.utils.conversions import to_body_frame

data_in_wb_bf = to_body_frame(data_in_wb)

# %%
# We can initialize the pipeline and call the ``calculate`` method similar to the individual algorithms.
# Here, we provide the algorithms the data, stride list, the sampling rate of the measurement. The ``turn_list``
# isn't available (or no turns), so the default value is used. Additionally, the ``replicate_matlab`` argument is used
# its default True value.
# See :class:`.MobilisedSDMO` for default parameters and details.
from mobgap.signal_based import MobilisedSDMO

sdmo = MobilisedSDMO()

sdmo.calculate(
    data=data_in_wb_bf,
    stride_list=reference_strides,
    sampling_rate_hz=short_trial.sampling_rate_hz,
)

# %%
# We get the signal-based parameters that can be calculated depending on the availability of the inputs.
# This output can be used within the gait sequence iterator to append results of each walking bout.
sdmo.signal_based_parameters


# %%
# Note that the ``calculate`` method raises warning regarding the availability of stride parameters and this
# results in 46 parameters calculated in total.
# We miss three more parameters because the expected stride list parameters
# (['stride_length_m', 'cadence_spm', 'stride_duration_s']) are given with a different name or not available in
# the reference list. We can rename and calculate the missing ones.
reference_strides = reference_strides.rename(
    columns={"duration_s": "stride_duration_s", "length_m": "stride_length_m"}
)
reference_strides["cadence_spm"] = (
    60 * reference_strides["speed_mps"] / reference_strides["stride_length_m"]
)

sdmo.calculate(
    data=data_in_wb_bf,
    stride_list=reference_strides,
    sampling_rate_hz=short_trial.sampling_rate_hz,
)
sdmo.signal_based_parameters
