# Import useful modules and packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from gaitmap.trajectory_reconstruction.orientation_methods import MadgwickAHRS
from gaitmap.utils.rotations import rotate_dataset_series
from mobgap.data import LabExampleDataset

fs = 100 # sampling frequency (Hz)
example_data = LabExampleDataset(reference_system="INDIP", reference_para_level="wb")  # alternatively: "StereoPhoto"
single_test = example_data.get_subset(cohort="HA", participant_id="001", test="Test11", trial="Trial1")
imu_data = single_test.data["LowerBack"]
LBh_cm = single_test.participant_metadata['SensorHeight'] # Sensor height (cm)
LBh = LBh_cm/100 # Sensor height (m)
# Reference parameters of a single GS
gs_i = 1 # first GS
reference_wbs = single_test.reference_parameters_.wb_list.loc[gs_i]
reference_ics = single_test.reference_parameters_.ic_list.loc[gs_i]
ics = reference_ics['ic']
reference_sl = single_test.reference_parameters_.stride_parameters['length_m'].loc[gs_i]
# segment signal of current gs
start = reference_wbs['start']
end = reference_wbs['end']
# [acc_x acc_y acc_z gyr_x gyr_y gyr_z]
imu_data_gs = imu_data[start:end]
# plot raw data over gait sequence
fig, axs = plt.subplots(2)
fig.tight_layout(pad=5.0)
fig.suptitle('Raw data')
# Acceleration
for i in imu_data_gs.columns[0:3]:
    axs[0].plot(imu_data_gs[i], label=i)
axs[0].set(xlabel='Samples', ylabel='(m/s^2)')
axs[0].set_title('Acceleration')
axs[0].legend()
# Angular velocity
for i in imu_data_gs.columns[3:]:
    axs[1].plot(imu_data_gs[i], label=i)
axs[1].set(xlabel='Samples', ylabel='(dps)')
axs[1].set_title('Angular rate')
axs[1].legend()
# Displaying the plot
plt.show()

# Sensor alignment
sampling_rate_hz = 100
rotated_imu_data = rotate_dataset_series(imu_data_gs, MadgwickAHRS(beta = 0.1).estimate(imu_data_gs, sampling_rate_hz=sampling_rate_hz).orientation_object_[:-1])

# plot rotated data over gait sequence
fig, axs = plt.subplots(2)
fig.tight_layout(pad=5.0)
fig.suptitle('Rotated data')
# Acceleration
for i in rotated_imu_data.columns[0:3]:
    axs[0].plot(rotated_imu_data[i], label=i)
axs[0].set(xlabel='Samples', ylabel='(m/s^2)')
axs[0].set_title('Acceleration')
axs[0].legend()
# Angular velocity
for i in rotated_imu_data.columns[3:]:
    axs[1].plot(rotated_imu_data[i], label=i)
axs[1].set(xlabel='Samples', ylabel='(dps)')
axs[1].set_title('Angular rate')
axs[1].legend()
# Displaying the plot
plt.show()
# rotating data does not seem to change anything in the signals
acc = imu_data_gs[['acc_x','acc_y','acc_z']]
# gyroscope
gyr = imu_data_gs[['gyr_x','gyr_y','gyr_z']]

# FILTERING
vacc=9.8*rotated_imu_data['acc_x']
fc=0.1
[df,cf] = butter(4,fc/(fs/2),'high')
# Note: the Matlab implementation actually makes use of filter rather than filtfilt.
# However, in the context of dedrifting, zero-phase distortion is preferred
# because it maintains the temporal relationships between different parts
# of the signal and preserves timing and phase information.
vacc_high=filtfilt(df,cf,vacc)

HSsamp= ics - start
K = 4.587

def zjilsV3(LB_vacc_high, fs, K, HSsamp, LBh):
    # step length estimation using the biomechanical model propose by Zijlstra & Hof
    # Zijlstra, W., & Hof, A. L. (2003). Assessment of spatio-temporal gait parameters from trunk accelerations during human walking.
    # Gait & posture, 18(2), 1-10.
    # Inputs:
    # - LB_vacc_high: vertical acceleration recorded on lower back, high-pass filtered
    # - fs: sampling frequency of input data (acc signal)
    # - model: contains the correction factor 'K' estimated by data from various clinical populations (training data)
    # - HSsamp: vector containing the timing of heal strikes (or initial contacts) events (in samples)
    # - LBh: Low Back height, i.e., the distance from ground to sensor location on lower back (in cm)
    #
    # Output:
    # - sl_zjilstra_v3: estimated step length.

    # vspeed calculation
    vspeed = -np.cumsum(LB_vacc_high) / fs
    # drift removal (high pass filtering)
    # Note: the Matlab implementation actually makes use of filter rather than filtfilt.
    # However, in the context of dedrifting, zero-phase distortion is preferred
    # because it maintains the temporal relationships between different parts
    # of the signal and preserves timing and phase information.
    fc = 1
    b, a = butter(4, fc / (fs / 2), 'high')
    speed_high = filtfilt(b, a, vspeed)
    # estimate vertical displacement
    vdis_high_v2 = np.cumsum(speed_high) / fs
    # initialize the output array as an array of zeros having one less element
    # than the array of ICs
    h_jilstra_v3 = np.zeros(len(HSsamp) - 1)
    # initial estimates of the stride length
    for k in range(len(HSsamp) - 1):
        # the k-th stride length value is initially estimated as
        # the absolute difference between the maximum and the minimum (range)
        # of the vertical displacement between the k-th and (k+1)-th IC
        h_jilstra_v3[k] = np.abs(max(vdis_high_v2[HSsamp[k]:HSsamp[k + 1]]) -
                                  min(vdis_high_v2[HSsamp[k]:HSsamp[k + 1]]))
    # Correction factor depending on population
    # To ask: how should that be passed to the function?
    # K = 4.587
    sl_zjilstra_v3 = K * np.sqrt(np.abs((2 * LBh * h_jilstra_v3) - (h_jilstra_v3 ** 2)))

    return sl_zjilstra_v3

sl_zjilstra_v3 = zjilsV3(vacc_high, fs, K, HSsamp, LBh)
sl_zjilstra_v3