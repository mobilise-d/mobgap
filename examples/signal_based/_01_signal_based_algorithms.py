"""
Signal-based parameters
=======================

This example shows how to use the :class:`.BaseSDMOCalculator` algorithms for calculating signal-based parameters.

First we load some data.

Example Data
------------
We load example data from the lab dataset together with the INDIP reference system.
We will use a single short-trail from the "HA" participant for this example, as it only contains a single gait sequence.
The signal-based parameters are calculated for each walking bout in the Mobilise-D pipelines, however, the algorithms
are designed to work with any signal window (although note that some parameters require to have some characteristics such
as certain number of strides).

"""

from mobgap.data import LabExampleDataset

lab_example_data = LabExampleDataset(reference_system="INDIP")
short_trial = lab_example_data.get_subset(
    cohort="HA", participant_id="001", test="Test5", trial="Trial2"
)

# %%
# SDMO
# ----
# To demonstrate the usage of any algorithms (:class:`.BaseSDMOCalculator`), we use the detected walking bout from the
# reference system as input.
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
# Then we can initialize a representative algorithm and call the ``calculate`` method.
# Although there might be some additional parameters in the algorithm init functions, the calculate methods are provided
# **kwargs so that, they can all be called and executed in the same manner. Note that you must explicitly provide
# ``data`` argument to all algorithms.
# For example, the :class:`.HarmonicRatio` algorithm is presented below that requires ``acc_columns`` in init, and ``data``,
# ``stride_list`` and ``sampling_rate_hz`` in the calculate method.

from mobgap.signal_based import HarmonicRatio

hr = HarmonicRatio(acc_columns=["acc_is", "acc_pa"])

hr.calculate(
    data=data_in_wb_bf,
    stride_list=reference_strides,
    sampling_rate_hz=short_trial.sampling_rate_hz,
)

# %%
# The output (``signal_based_parameters``) attribute contains the calculated parameters.
hr.signal_based_parameters

# %%
# Another example is the :class:`.SDRange` algorithm. This algorithm doesn't require any arguments in the init, and
# only uses the ``data``.

from mobgap.signal_based import SDRange

sdr = SDRange()

sdr.calculate(
    data=data_in_wb_bf,
    stride_list=reference_strides,
    sampling_rate_hz=short_trial.sampling_rate_hz,
)

# %%
# The output (``signal_based_parameters``) attribute contains the calculated parameters.
sdr.signal_based_parameters
