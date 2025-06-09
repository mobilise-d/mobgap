"""
SL Zijlstra
===========

This example shows how to use the Zijlstra algorithm for calculating stride length
and how its results compare to the original matlab implementation.

Below we will demonstrate the usage of both methods. But first we load some data.

Example Data
------------
We load example data from the lab dataset together with the INDIP reference system.
We will use a single short-trail from the "HA" participant for this example, as it only contains a single gait sequence.
The stride length algorithm is designed to work on a single gait sequence at a time.

"""

from mobgap.data import LabExampleDataset

lab_example_data = LabExampleDataset(reference_system="INDIP")
short_trial = lab_example_data.get_subset(
    cohort="HA", participant_id="001", test="Test5", trial="Trial2"
)

# %%
# SlZijlstra
# ----------
# To demonstrate the usage of :class:`~mobgap.stride_length.SlZijlstra` we use the detected initial contacts from the
# reference system as input.
reference_ic = short_trial.reference_parameters_relative_to_wb_.ic_list
reference_ic

# %%
reference_gs = short_trial.reference_parameters_relative_to_wb_.wb_list
reference_gs

# %%
# We only pick the first gait sequence for this example.
gs_id = reference_gs.index[0]
data_in_gs = short_trial.data["LowerBack"].iloc[
    reference_gs.start.iloc[0] : reference_gs.end.iloc[0]
]
ics_in_gs = reference_ic.loc[gs_id]
sensor_height = short_trial.participant_metadata["sensor_height_m"]

# %%
# Like most algorithms, the SlZijlstra requires the data to be in body frame coordinates.
# As we know the sensor was well aligned, we can just use ``to_body_frame`` to transform the data.
from mobgap.utils.conversions import to_body_frame

data_in_gs_bf = to_body_frame(data_in_gs)

# %%
# Then we initialize the algorithm and call the ``calculate`` method.
# We use the scaling parameters optimized on the MSProject dataset and leave all other values as default.
from mobgap.stride_length import SlZijlstra

sl_zijlstra = SlZijlstra(
    **SlZijlstra.PredefinedParameters.step_length_scaling_factor_ms_ms
)

sl_zijlstra.calculate(
    data=data_in_gs_bf,
    initial_contacts=ics_in_gs,
    sensor_height_m=sensor_height,
    sampling_rate_hz=short_trial.sampling_rate_hz,
)

# %%
# We get an output that contains the stride length for each second of the gaits sequence.
# The index represents the sample of the center of the second the stride length value belongs to.
sl_zijlstra.stride_length_per_sec_

# %%
# To show that the approach results in roughly the "correct" stride length value, we can compare the average stride
# length to the reference system.
reference_sl = reference_gs["avg_stride_length_m"].loc[gs_id]
zijlstra_avg_sl = sl_zijlstra.stride_length_per_sec_["stride_length_m"].mean()
print("Sensor frame:")
print(f"Average stride length from reference: {reference_sl:.3f} m")
print(f"Calculated average per-sec stride length: {zijlstra_avg_sl:.3f} m")

# %%
# In addition, to this primary output, that is available for all stride length algorithms, the Zijlstra algorithm also
# provides some of the intermediate results that are used to calculate the stride length values.
# These might be helpful to generate further insides into the data or debug results.
#
# First it provides the step length values for each detected step, which is an annotated version of the input
# initial contacts.
sl_zijlstra.raw_step_length_per_step_

# %%
# Second, it provides the step length values for each second of the gait sequence.
# This is basically just half of the stride length values.
sl_zijlstra.step_length_per_sec_

# %%
# Working in the global frame
# ---------------------------
# The algorithm assume that the accelerometer axes are aligned with the global reference system.
# I.e. that ``acc_x`` aligns with the vertical direction throughout the entire measurement.
# However, this assumption is not practical for the realworld measurement setup, even if an initial alignment is
# performed.
# The pelvis movement will always result in some changes in the orientation of the sensor.
# To overcome this, we can use a sensor orientation method to correct the sensor orientation to project the acceleration
# values to the global reference system.
#
# This is rarely perfect and might introduce some additional errors, but might help to improve the results overall.
# There are two ways to achieve this:
#
# 1. You could use an orientation method to correct the sensor orientation to the global reference system on the entire
#    recording.
#    This can often result in better orientation estimations, as the algorithm has time to converge, but might be
#    computationally expensive.
# 2. Alternatively, you could use the ``orientation_method`` argument of ``SlZijlstra`` to apply a global frame
#    transform only on the provided data.
#    We will show this case below.
#
# We basically do the same above, but we correct the sensor orientation to the global frame using a Madgwick
# complementary filter, to show the improvement of performing re-orientation.
from mobgap.orientation_estimation._madgwick import MadgwickAHRS

sl_zijlstra_reoriented = SlZijlstra(
    orientation_method=MadgwickAHRS(beta=0.2),
    **SlZijlstra.PredefinedParameters.step_length_scaling_factor_ms_ms,
)
sl_zijlstra_reoriented.calculate(
    data=data_in_gs_bf,
    initial_contacts=ics_in_gs,
    sensor_height_m=sensor_height,
    sampling_rate_hz=short_trial.sampling_rate_hz,
)
# %%
# As before, we get an output that contains the stride length for each second of the gaits sequence.
# The index represents the sample of the center of the second the stride length value belongs to.
sl_zijlstra_reoriented.stride_length_per_sec_

# %%
# To show that the approach results in roughly the "correct" stride length value, we can compare the average stride
# length to the reference system.
zijlstra_reoriented_avg_sl = sl_zijlstra_reoriented.stride_length_per_sec_[
    "stride_length_m"
].mean()
print("")
print("Global frame:")
print(f"Average stride length from reference: {reference_sl:.3f} m")
print(
    f"Calculated average per-sec stride length: {zijlstra_reoriented_avg_sl:.3f} m"
)

# %%
# Note, that in this case, the results did not really change between the version with and without Madgwick.
# However, this is not a general statement, but heavily depends on the data characteristics.
