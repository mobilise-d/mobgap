# Import useful modules and packages
import matplotlib.pyplot as plt
import numpy as np
from hampel import hampel
import pandas as pd
from sklearn.decomposition import PCA
from scipy.signal import butter, filtfilt, lfilter, medfilt
from gaitmap.trajectory_reconstruction.orientation_methods import MadgwickAHRS
from gaitmap.utils.rotations import rotate_dataset_series
from mobgap.data import LabExampleDataset

sampling_rate_hz = 100 # sampling frequency (Hz)
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
duration = np.floor(end/sampling_rate_hz) - np.floor(start/sampling_rate_hz) # bottom-rounded duration (s)

# [acc_x acc_y acc_z gyr_x gyr_y gyr_z]
imu_data_gs = imu_data[start:end]
acc = imu_data_gs[['acc_x','acc_y','acc_z']]/9.81
gyr = imu_data_gs[['gyr_x','gyr_y','gyr_z']]
# plot raw acceleration over gait sequence
plt.plot(acc,label = ['acc_x','acc_y','acc_z'])
plt.title('Raw data')
plt.ylabel('Acceleration (m/s^2)')
plt.xlabel('Samples')
plt.legend()
# Displaying the plot
plt.show()

# Sensor alignment
rotated_imu_data = rotate_dataset_series(imu_data_gs, MadgwickAHRS(beta = 0.1).estimate(imu_data_gs, sampling_rate_hz=sampling_rate_hz).orientation_object_[:-1])
pca = PCA(n_components=3)
pca.fit(acc)
pcaCoef = pca.components_
newAcc = np.matmul(acc.to_numpy(), pcaCoef)
newAcc = pd.DataFrame(newAcc, columns = ['acc_x', 'acc_y', 'acc_z'])
# plot aligned acceleration over gait sequence
plt.plot(newAcc,label = ['acc_x','acc_y','acc_z'])
plt.title('Aligned data')
plt.ylabel('Acceleration (m/s^2)')
plt.xlabel('Samples')
plt.legend()
# Displaying the plot
plt.show()
# rotating data does not seem to change anything in the signals

# FILTERING
vacc=9.8 * newAcc['acc_x']
fc=0.1
[df,cf] = butter(4,fc/(sampling_rate_hz/2),'high')
# Note: the Matlab implementation actually makes use of filter rather than filtfilt.
# However, in the context of dedrifting, zero-phase distortion is preferred
# because it maintains the temporal relationships between different parts
# of the signal and preserves timing and phase information.
vacc_high=lfilter(df,cf,vacc)

plt.plot(vacc, label = 'raw')
plt.plot(vacc_high, label = 'filtered')
plt.legend()
plt.show()

HSsamp= ics - start
HStime = HSsamp/sampling_rate_hz
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
    speed_high = lfilter(b, a, vspeed)
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

sl_zjilstra_v3 = zjilsV3(vacc_high, sampling_rate_hz, K, HSsamp, LBh)


def stride2sec(ICtime, duration, stl):
# if the number of SL values is lower than the one of ICs, replicate the last element of the SL array until they have the
# same length.
    if len(stl) < len(ICtime):
        stl = np.concatenate([stl, np.tile(stl[-1], len(ICtime) - len(stl))])
        # stl = np.concatenate([stl, np.tile(stl[-1], (len(ICtime) - len(stl), 1))])
    # hampel filter: For each sample of stl, the function computes the
    # median of a window composed of the sample and its four surrounding
    # samples, two per side. It also estimates the standard deviation
    # of each sample about its window median using the median absolute
    # deviation. If a sample differs from the median by more than three
    # standard deviations, it is replaced with the median.

    # stl = medfilt(stl, kernel_size=3)
    stl = hampel(stl, window_size=2).filtered_data
    N = int(np.floor(duration)) # greater integer number of seconds included in the WB
    winstart = np.arange(1, N + 1) - 0.5 # start of each second
    winstop = np.arange(1, N + 1) + 0.5 # end of each second

    stSec = np.zeros(N) # initialize array of SL values per second

    for i in range(N): # consider each second
        if winstop[i] < ICtime[0]:
            stSec[i] = -1 # set SL value to -1 if current sec ends before the first IC occurs
        elif winstart[i] > ICtime[-1]:
            stSec[i] = -2 # set SL value to -2 if current sec starts after the last IC occurs
        else: # if current sec is between the first and the last IC
            ind = (winstart[i] <= ICtime) & (ICtime <= winstop[i]) # find indices of ICs that are included in the current sec.
            if np.sum(ind) == 0: # if there are no ICs in the current sec...
                inx = winstart[i] >= ICtime # indices of ICs that occur before sec starts
                aa = stl[np.logical_or(np.abs(np.diff(inx)), False)] # take first SL value before current second starts
                iny = ICtime >= winstop[i] # indices of ICs that occur after sec ends
                bb = stl[np.logical_or(False, np.abs(np.diff(iny)))] # take first SL value after current second ends
                stSec[i] = (aa + bb) / 2 # the SL value of the current second is the average of aa and bb
            else: # if there are one or more ICs in the current sec
                stSec[i] = np.nanmean(stl[ind]) # the SL value of the current sec is the average all SL values in the current sec

    myInx = stSec == -1 # indices of seconds that end before the first IC occurs (empty seconds)
    tempax = np.arange(0, N) # array of seconds
    tempax2 = tempax[~myInx] # indices of seconds that end AFTER the first IC occurs
    stSec[myInx] = stSec[tempax2[0]] # set SL values of empty seconds to the first SL value of non-empty seconds

    myInd = stSec == -2 # indices of seconds that start after the last IC occurs (empty seconds)
    tempax3 = tempax[~myInd] # indices of seconds that start BEFORE the last IC occurs
    stSec[myInd] = stSec[tempax3[-1]] # set SL values of empty seconds to the last SL value of non-empty seconds

    stSec = hampel(stSec, window_size=2).filtered_data # re-apply the same hampel filter to SL values per second
    # if the number of SL values is lower than the one of ICs,
    # replicate the last element of the SL array until they have the
    # same length
    if len(stSec) < duration:
        stSec = np.concatenate([stSec, np.tile(stSec[-1], (duration - len(stSec)))])
    # if the number of SL values is lower than the one of ICs,
    # truncates to the last element of the SL array that is included in duration
    elif len(stSec) > duration:
        stSec = stSec[:duration]
    return stSec
slSec_zjilstra_v3 = stride2sec(HStime.to_numpy(), duration, sl_zjilstra_v3)
slmat = slSec_zjilstra_v3[0:duration]