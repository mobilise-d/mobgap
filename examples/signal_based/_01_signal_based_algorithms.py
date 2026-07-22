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
    cohort="HA", participant_id="001", test="Test11", trial="Trial1"
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
# We may also use the turn data. We select the example data with turns.
reference_turns = (
    short_trial.reference_parameters_relative_to_wb_.turn_parameters
)
reference_turns

# %%
# We pick the walking bout for this example with turns.
wb_id = 2
reference_strides = reference_strides.loc[wb_id]
reference_turns = reference_turns.loc[wb_id]
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
# ``**kwargs`` so that they can all be called and executed in the same manner. Note that you must explicitly provide
# ``data`` argument to all algorithms.
# The output (``signal_based_parameters_``) attribute contains the calculated parameters.
# Below, we define the extra parameters and we provide examples to all algorithms.
params = dict(
    stride_list=reference_strides,
    sampling_rate_hz=short_trial.sampling_rate_hz,
    turn_list=reference_turns,
    replicate_matlab=True,
)

# %%
# the :class:`.TurnSDMO` algorithm calculates parameters related to the turning.

from mobgap.signal_based import TurnSDMO

turn = TurnSDMO()

turn.calculate(data=data_in_wb_bf, **params)

turn.signal_based_parameters_

# %%
# the :class:`.StrideLevelSDMO` algorithm calculates parameters related to the stride parameters.

from mobgap.signal_based import StrideLevelSDMO

stride_level = StrideLevelSDMO(stride_list_columns=["length_m", "duration_s"])

stride_level.calculate(data=data_in_wb_bf, **params)

stride_level.signal_based_parameters_


# %%
# the :class:`.RMS` algorithm.

from mobgap.signal_based import RMS

rms = RMS()

rms.calculate(data=data_in_wb_bf, **params)

rms.signal_based_parameters_

# %%
# the :class:`.RegularitySymmetry` algorithm.

from mobgap.signal_based import RegularitySymmetry

regularity_symmetry = RegularitySymmetry()

regularity_symmetry.calculate(data=data_in_wb_bf, **params)

regularity_symmetry.signal_based_parameters_


# %%
# the :class:`.FrequencyAmplitudeWidth` algorithm.

from mobgap.signal_based import FrequencyAmplitudeWidth

frequency_amplitude = FrequencyAmplitudeWidth(
    acc_columns=["acc_is", "acc_ml", "acc_pa"]
)

frequency_amplitude.calculate(data=data_in_wb_bf, **params)

frequency_amplitude.signal_based_parameters_


# %%
# the :class:`.SampleEntropy` algorithm.

from mobgap.signal_based import SampleEntropy

sample_entropy = SampleEntropy(dim=2, r=0.15, acc_columns=["acc_is"])

sample_entropy.calculate(data=data_in_wb_bf, **params)

sample_entropy.signal_based_parameters_


# %%
# the :class:`.HarmonicRatio` algorithm.

from mobgap.signal_based import HarmonicRatio

harmonic_ratio = HarmonicRatio(acc_columns=["acc_is", "acc_pa"])

harmonic_ratio.calculate(data=data_in_wb_bf, **params)

harmonic_ratio.signal_based_parameters_


# %%
# the :class:`.SDRange` algorithm.

from mobgap.signal_based import SDRange

sd_range = SDRange()

sd_range.calculate(data=data_in_wb_bf, **params)

sd_range.signal_based_parameters_


# %%
# the :class:`.Jerk` algorithm.

from mobgap.signal_based import Jerk

jerk = Jerk(
    acc_columns=["acc_is", "acc_ml", "acc_pa"],
)

jerk.calculate(data=data_in_wb_bf, **params)

jerk.signal_based_parameters_

# %%
# the :class:`.AngularAcceleration` algorithm.

from mobgap.signal_based import AngularAcceleration

angular_acceleration = AngularAcceleration(
    gyr_columns=["gyr_is", "gyr_ml", "gyr_pa"]
)

angular_acceleration.calculate(data=data_in_wb_bf, **params)

angular_acceleration.signal_based_parameters_
