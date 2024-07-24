r"""
.. _cad_from_ic:

Cadence from Initial Contacts
=============================

The most obvious way to calculate cadence is to use the detected initial contacts.
However, initial contact (IC) detection from a single lower back sensor is never perfect.
If we naively calculate cadence via a step time derived by "diffing" the initial contacts, missing ICs or
even actual breaks in the gait sequence will lead to wrong cadence values (likely underestimating the cadence).
We need to be robust to these types of errors.

On the other hand, for the cadence estimation it is less important that the position of the IC within the
stride is perfectly correct.
This means, different IC detection methods might be optimal for the cadence estimation than for the IC detection itself.

With these two things in mind, we implemented "Proxy" cadence algorithms that work on top of any IC detection method.
Two variants exist:

1. :class:`~mobgap.algorithm.CadFromIc` calculates the cadence directly from the provided ICs using step-to-step
   smoothing to deal with missing ICs and breaks in the gait sequence.
   This method should be used, if you want to use the same IC detection method for the cadence estimation than for the
   IC detection itself (i.e. we are not rerunning any calculations, we just use the provided ICs).
2. :class:`~mobgap.algorithm.CadFromIcDetector` is a proxy algorithm that wraps around any IC detection algorithm.
   Compared to the previous method, this method will ignore the provided ICs and run the IC detection algorithm it
   wraps (which can be different from the one used for the IC detection itself) to find new ICs that are only used
   for the Cadence estimation.

Both methods than use step-to-step smoothing to deal with missing ICs and breaks in the gait sequence and then
interpolate all step time values to seconds to provide an average cadence for each one second interval of the
recording.

Below we will demonstrate the usage of both methods. But first we load some data.

Example Data
------------
We load example data from the lab dataset together with the INDIP reference system.
We will use a single short-trail from the "HA" participant for this example, as it only contains a single gait sequence.
All the cadence algorithms are designed to work on a single gait sequence at a time.

"""

from mobgap.data import LabExampleDataset

lab_example_data = LabExampleDataset(reference_system="INDIP")
short_trial: LabExampleDataset = lab_example_data.get_subset(
    cohort="HA", participant_id="001", test="Test5", trial="Trial2"
)

# %%
# CadFromIc
# ---------
# To demonstrate the usage of :class:`~mobgap.algorithm.CadFromIc` we use the detected initial contacts from the
# reference system as input.
reference_ic = short_trial.reference_parameters_relative_to_wb_.ic_list
reference_ic

# %%
reference_gs = short_trial.reference_parameters_relative_to_wb_.wb_list
reference_gs

# %%
# Then we initialize the algorithm and call the ``calculate`` method.
# Note that we use the ``sampling_rate_hz`` of the actual data and not the reference system.
# This is because, the reference parameters are already converted to the data sampling rate.
from mobgap.cadence import CadFromIc

cad_from_ic = CadFromIc()

gs_id = reference_gs.index[0]
data_in_gs = short_trial.data_ss.iloc[
    reference_gs.start.iloc[0] : reference_gs.end.iloc[0]
]
ics_in_gs = reference_ic[["ic"]].loc[gs_id]

cad_from_ic.calculate(
    data_in_gs,
    initial_contacts=ics_in_gs,
    sampling_rate_hz=short_trial.sampling_rate_hz,
)

# %%
# We get an output that contains the cadence for each second of the gaits sequence.
# The index represents the sample of the center of the second the cadence value belongs to.
#
# Note, that there might be NaNs in the output, if there are breaks in the gait sequence and further, that the final
# second might only be partially covered by the gait sequence.
# We still calculate a cadence value for the entire second, but it is only based on the steps that occur in the part
# that was actually covered by the gait sequence.
cad_from_ic.cadence_per_sec_

# %%
# To show that the approach results in roughly the "correct" cadence value, we can compare the average cadence to the
# reference system.
reference_cad = reference_gs["avg_cadence_spm"].loc[gs_id]
cad_from_ic_avg_cad = cad_from_ic.cadence_per_sec_["cadence_spm"].mean()
print(f"Average stride cadence from reference: {reference_cad:.2f} steps/min")
print(
    f"Calculated average per-sec cadence: {cad_from_ic_avg_cad:.2f} steps/min"
)

# %%
# Note that if we would have breaks in the gait sequence, the method would try to interpolate the cadence values for
# short breaks, but would provide NaNs for longer breaks.
# This is controlled by the ``max_interpolation_gap_s`` parameter.
#
# CadFromIcDetector
# -----------------
# For the :class:`~mobgap.cadence.CadFromIcDetector` we need to supply an IC detection algorithm.
# In this case we use the :class:`~mobgap.initial_contacts.IcdShinImproved` algorithm.
# We could also use any other IC detection algorithm or adapt the parameters of the IC detection algorithm.
from mobgap.cadence import CadFromIcDetector
from mobgap.initial_contacts import IcdShinImproved

cad_from_ic_detector = CadFromIcDetector(IcdShinImproved())

# %%
# Now we can call the ``calculate`` method with the same data as before.
# Note, that we are still passing the initial contacts from the "previous" calculation step to fulfill the API.
# However, internally the algorithm will ignore the provided ICs and rerun the IC detection using the provided IC
# detector.
#
# Note, that we need to convert the data to the body frame, as the underlying IC-detector requires it.
from mobgap.utils.conversions import to_body_frame

cad_from_ic_detector.calculate(
    to_body_frame(data_in_gs),
    initial_contacts=ics_in_gs,
    sampling_rate_hz=short_trial.sampling_rate_hz,
)

# %%
# .. note:: By default the ``CadFromIcDetector`` will raise a warning to inform the user that the passed ICs are
#           ignored.
#           This is intentionally, as we assume first time users might be confused by that fact.
#           If you are aware of this and want to get rid of the warning, you can set ``silence_ic_warning`` to
#           True on the ``CadFromIcDetector`` object.
#
# We get the same output structure as before.
cad_from_ic_detector.cadence_per_sec_

# %%
# But we can also access the detected initial contacts.
cad_from_ic_detector.internal_ic_list_

# %%
# Or the entire detector object.
cad_from_ic_detector.ic_detector_

# %%
# To show that the approach results in roughly the "correct" cadence value, we can compare the average cadence to the
# reference system.
#
# .. note:: Compared to the previous method, the cadence value are more different, as we actually run a IC
#          detection algorithm to find the ICs and not just used the values provided by the reference.
cad_from_ic_detector_avg_cad = cad_from_ic_detector.cadence_per_sec_[
    "cadence_spm"
].mean()
print(f"Average stride cadence from reference: {reference_cad:.2f} steps/min")
print(
    f"Calculated average per-sec cadence (CadFromIC): {cad_from_ic_avg_cad:.2f} steps/min"
)
print(
    f"Calculated average per-sec cadence (CadFromIcDetector): {cad_from_ic_detector_avg_cad:.2f} steps/min"
)
