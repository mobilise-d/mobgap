"""
SL Zijlstra
=============================

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
short_trial: LabExampleDataset = lab_example_data.get_subset(
    cohort="HA", participant_id="001", test="Test5", trial="Trial2"
)

# %%
# SlZijlstra
# ---------
# To demonstrate the usage of :class:`~mobgap.algorithm.SlZijlstra` we use the detected initial contacts from the
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
from gaitmap.trajectory_reconstruction.orientation_methods import MadgwickAHRS

from mobgap.sl import SlZijlstra

# Initially, we apply the algorithm without re-orienting the sensor frame to the global frame
MS_MS_params = (
    SlZijlstra.PredefinedParameters.step_length_scaling_factor_ms_ms
)  # default predefined parameters for the biomechanical model
sl_zijlstra = SlZijlstra(**MS_MS_params)

gs_id = reference_gs.index[0]
data_in_gs = short_trial.data["LowerBack"].iloc[reference_gs.start.iloc[0]: reference_gs.end.iloc[0]]
ics_in_gs = reference_ic[["ic"]].loc[gs_id]  # reference initial contacts
sensor_height = short_trial.participant_metadata["SensorHeight"] / 100  # sensor height (cm)

sl_zijlstra.calculate(
    data=data_in_gs,
    initial_contacts=ics_in_gs,
    sensor_height_m=sensor_height,
    sampling_rate_hz=short_trial.sampling_rate_hz,
)

# %%
# We get an output that contains the stride length for each second of the gaits sequence.
# The index represents the sample of the center of the second the stride length value belongs to.
sl_zijlstra.stride_length_per_sec_list_

# %%
# To show that the approach results in roughly the "correct" stride length value, we can compare the average stride
# length to the reference system.
reference_sl = reference_gs["avg_stride_length_m"].loc[gs_id]
zijlstra_avg_sl = sl_zijlstra.stride_length_per_sec_list_["stride_length_m"].mean()
print("Sensor frame:")
print(f"Average stride length from reference: {reference_sl:.2f} m")
print(f"Calculated average per-sec stride length: {zijlstra_avg_sl:.2f} m")

"""
The algorithm assume that the accelerometer axes are aligned with the global reference system, and/or
the measurement setup includes functional calibration procedures. However, this assumption is not practical 
for the realworld measurement setup. The alternative solution is to take advantage of a 3D gyroscope 
available in the IMU devices and correct the sensor orientation using complementary filters like Madgwick.
"""
# Now we do the same thing but we correct the sensor orientation to the global frame using a Madgwick
# complementary filter, to show the improvement of performing re-orientation.
sl_zijlstra_reoriented = SlZijlstra(orientation_method=MadgwickAHRS(beta=0.2), **MS_MS_params)
sl_zijlstra_reoriented.calculate(
    data=data_in_gs,
    initial_contacts=ics_in_gs,
    sensor_height_m=sensor_height,
    sampling_rate_hz=short_trial.sampling_rate_hz,
)
# %%
# We get an output that contains the stride length for each second of the gaits sequence.
# The index represents the sample of the center of the second the stride length value belongs to.
sl_zijlstra_reoriented.stride_length_per_sec_list_

# %%
# To show that the approach results in roughly the "correct" stride length value, we can compare the average stride
# length to the reference system.
zijlstra_reoriented_avg_sl = sl_zijlstra_reoriented.stride_length_per_sec_list_["stride_length_m"].mean()
print("")
print("Global frame:")
print(f"Average stride length from reference: {reference_sl:.2f} m")
print(f"Calculated average per-sec stride length: {zijlstra_reoriented_avg_sl:.2f} m")
