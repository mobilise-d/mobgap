"""
Secondary outcome calculators
=============================

This example shows how to use the
:class:`~mobgap.secondary_outcomes.base.BaseSecondaryOutcomeCalculator` algorithms to calculate secondary outcomes.

First we load some data.

Example Data
------------
We load example data from the lab dataset together with the INDIP reference system.
We will use a single short-trail from the "HA" participant for this example, as it only contains a single gait sequence.
The algorithms operate on prepared signal windows after primary analysis. Some parameters require additional inputs,
such as retained stride boundaries, or a minimum amount of data.

"""

from mobgap.data import LabExampleDataset

lab_example_data = LabExampleDataset(reference_system="INDIP")
short_trial = lab_example_data.get_subset(
    cohort="HA", participant_id="001", test="Test11", trial="Trial1"
)

# %%
# SDMO
# ----
# To demonstrate the usage of any algorithms (:class:`~mobgap.secondary_outcomes.base.BaseSecondaryOutcomeCalculator`), we use the detected walking bout from the
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
wb = short_trial.reference_parameters_relative_to_wb_.wb_list.loc[wb_id]
data_in_wb = short_trial.data["LowerBack"].iloc[wb.start : wb.end]

# %%
# The data is required to be in body frame coordinates.
# As we know the sensor was well aligned, we can just use ``to_body_frame`` to transform the data.
from mobgap.utils.conversions import to_body_frame

data_in_wb_bf = to_body_frame(data_in_wb)

# %%
# Then we can initialize a representative algorithm and call the ``calculate`` method.
# The base calculator contract names the inputs used by the shared WB call. Each calculator consumes the inputs it
# needs and accepts the others through ``**kwargs``. Direct calls can supply only the concrete calculator's inputs.
# The output ``signal_based_parameters_`` is a Series containing all scalar parameters for one prepared window.
# Below, we define the extra parameters and we provide examples to all algorithms.
params = dict(
    stride_list=reference_strides,
    sampling_rate_hz=short_trial.sampling_rate_hz,
)

# %%
# the :class:`~mobgap.secondary_outcomes.TurnKinematics` algorithm calculates parameters related to the turning.

from mobgap.secondary_outcomes import TurnKinematics

turn = TurnKinematics()

# Select one turn window explicitly. Metadata and iteration over turns belong to the caller.
first_turn = reference_turns.iloc[0]
turn.calculate(
    data=data_in_wb_bf.iloc[int(first_turn.start) : int(first_turn.end)],
    sampling_rate_hz=short_trial.sampling_rate_hz,
)

turn.signal_based_parameters_

# %%
# the :class:`~mobgap.secondary_outcomes.StrideParameterCV` algorithm calculates parameters related to the stride parameters.

from mobgap.secondary_outcomes import StrideParameterCV

stride_level = StrideParameterCV(
    stride_list_columns=[
        "stride_duration_s",
        "stride_length_m",
        "cadence_spm",
        "speed_mps",
    ]
)

stride_level.calculate(data=data_in_wb_bf, **params)

stride_level.signal_based_parameters_


# %%
# the :class:`~mobgap.secondary_outcomes.RootMeanSquare` algorithm.

from mobgap.secondary_outcomes import RootMeanSquare

rms = RootMeanSquare(
    acc_columns=["acc_is", "acc_ml", "acc_pa"],
    gyr_columns=["gyr_is", "gyr_ml", "gyr_pa"],
)

rms.calculate(data=data_in_wb_bf, **params)

rms.signal_based_parameters_

# %%
# the :class:`~mobgap.secondary_outcomes.GaitRegularityAndSymmetry` algorithm.

from mobgap.secondary_outcomes import GaitRegularityAndSymmetry

# This class uses an odd centered mean with symmetric shrinking edge windows. To use the nominal span with
# truncated edge windows instead, select :class:`~mobgap.secondary_outcomes.GaitRegularityAndSymmetryTruncatedMean`.
regularity_symmetry = GaitRegularityAndSymmetry()

regularity_symmetry.calculate(data=data_in_wb_bf, **params)

regularity_symmetry.signal_based_parameters_


# %%
# the :class:`~mobgap.secondary_outcomes.SpectralPeakFeatures` algorithm.

from mobgap.secondary_outcomes import SpectralPeakFeatures

frequency_amplitude = SpectralPeakFeatures(
    acc_columns=["acc_is", "acc_ml", "acc_pa"]
)

frequency_amplitude.calculate(data=data_in_wb_bf, **params)

frequency_amplitude.signal_based_parameters_


# %%
# the :class:`~mobgap.secondary_outcomes.SampleEntropy` algorithm.

from mobgap.secondary_outcomes import SampleEntropy

sample_entropy = SampleEntropy(dim=2, r=0.15, acc_columns=["acc_is"])

sample_entropy.calculate(data=data_in_wb_bf, **params)

sample_entropy.signal_based_parameters_


# %%
# the :class:`~mobgap.secondary_outcomes.HarmonicRatioPeakTrough` algorithm.

from mobgap.secondary_outcomes import HarmonicRatioPeakTrough

harmonic_ratio = HarmonicRatioPeakTrough(acc_columns=["acc_is", "acc_pa"])

harmonic_ratio.calculate(data=data_in_wb_bf, **params)

harmonic_ratio.signal_based_parameters_

# %%
# The separate Fourier method reads the same supplied stride intervals, uses actual Fourier amplitudes,
# and averages per-stride ratios. Its distinct output names allow both methods to be used together.
# The default WB calculator configuration uses the peak/trough estimator above.
from mobgap.secondary_outcomes import HarmonicRatioFourier

harmonic_ratio_fourier = HarmonicRatioFourier(
    acc_columns=["acc_is", "acc_pa", "acc_ml"]
)
harmonic_ratio_fourier.calculate(data=data_in_wb_bf, **params)
harmonic_ratio_fourier.signal_based_parameters_


# %%
# the :class:`~mobgap.secondary_outcomes.StandardDeviationAndRange` algorithm.

from mobgap.secondary_outcomes import StandardDeviationAndRange

sd_range = StandardDeviationAndRange()

sd_range.calculate(data=data_in_wb_bf, **params)

sd_range.signal_based_parameters_


# %%
# the :class:`~mobgap.secondary_outcomes.LogDimensionlessJerk` algorithm.

from mobgap.secondary_outcomes import LogDimensionlessJerk

log_dimensionless_jerk = LogDimensionlessJerk(
    acc_columns=["acc_is", "acc_ml", "acc_pa"],
    gyr_columns=["gyr_is", "gyr_ml", "gyr_pa"],
)

log_dimensionless_jerk.calculate(data=data_in_wb_bf, **params)

log_dimensionless_jerk.signal_based_parameters_

# %%
# the :class:`~mobgap.secondary_outcomes.JerkRMSRatio` algorithm.

from mobgap.secondary_outcomes import JerkRMSRatio

rms_jerk_ratio = JerkRMSRatio()

rms_jerk_ratio.calculate(data=data_in_wb_bf, **params)

rms_jerk_ratio.signal_based_parameters_
