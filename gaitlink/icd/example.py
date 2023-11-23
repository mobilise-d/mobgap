from gaitlink.data import LabExampleDataset
from gaitlink.data_transform._filter import EpflDedriftedGaitFilter
from gaitlink.data_transform._filter import EpflGaitFilter
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import cwt, ricker
from scipy.integrate import cumtrapz
from gaitlink.consts import GRAV_MS2
import numpy as np
# Specify target cohort, ID, test and trial
target_cohort = "MS"
target_ID = "001"
target_Test = "Test5"
target_Trial = "Trial1"
#%%
# Functions
def resamp_interp(y, fs_initial, fs_final):
    recordingTime = len(y)
    x = np.arange(1, recordingTime + 1)  # MATLAB uses 1-based indexing
    xq = np.arange(1, recordingTime + 1, fs_initial / fs_final)  # Create the new time vector
    interp_func = interp1d(x, y, kind='linear', axis=0, fill_value='extrapolate') # returns an interpolation function
    yResamp = interp_func(xq)
    return yResamp


def MaxPeaksBetweenZC(x):
    # peaks and locations from vector x between zero crossings and location
    # if x.shape[0] != 1:
    #     raise ValueError('x must be a row vector')
    # if x.size != len(x):
    #     raise ValueError('x must be a row vector')

    # Find zero crossing locations
    #ix = np.where(np.abs(np.diff(np.sign(x))) == 2)[0] + 1
    ix = np.asarray(np.abs(np.diff(np.sign(x))) == 2).nonzero()[0]+1

    L = len(ix) - 1  # Number of zero crossings minus 1

    # Find the indices of maximum values between zero crossings
    ipk = [imax(np.abs(x[ix[i]:ix[i+1]])) for i in range(L)]
    # the previous line basically considers L separate intervals between the L+1 zero crossings and returns the index of
    # the peak in each interval

    # Calculate the indices of the maximum values in the original vector
    ipks = [ix[i] + ipk[i] for i in range(L)]

    # Get the signed max/min values
    pks = x[ipks]

    return pks, ipks

def imax(x):
    # Return indices of maximum values
    return np.argmax(x)

#%% Load data
example_data = LabExampleDataset()
ha_example_data = example_data.get_subset(cohort=target_cohort) # select cohort
single_test = ha_example_data.get_subset(participant_id=target_ID, test=target_Test, trial=target_Trial)
# The raw IMU data of the LowerBack sensor:
imu_data = single_test.data["LowerBack"]
# vertical acceleration--> dimension: [g]
accV = imu_data["acc_x"]/GRAV_MS2
# %% Load reference system data
example_data_with_reference = LabExampleDataset(reference_system="Stereophoto")
single_trial_with_reference = example_data_with_reference.get_subset(
    cohort=target_cohort, participant_id=target_ID, test=target_Test, trial=target_Trial
)
#TODO: USE OUTPUT OF GSD (INSTANTS OF START AND END OF GSs) TO EXTRACT IMU DATA

# Sampling frequency of data
fs = single_trial_with_reference.sampling_rate_hz # [Hz]
# Re-sampling frequency
algorithm_target_fs=40 # [Hz]

# Get Start and End of gait sequences
# Start and End from gsd algo should be in [s] starting from 1/fs at the first sample
# (0 in Python, 1 in Matlab). Here, Start and End are taken from Stereophotogrammetry.
# Since Matlab starts counting samples from 1, Matlab events are subtracted 1 to be used
# as indices in Python.
s = round(single_trial_with_reference.reference_parameters_['lwb'][0]["Start"]*fs)-1
e = round(single_trial_with_reference.reference_parameters_['lwb'][0]["End"]*fs)-1
gs = accV.array[s:e+1].to_numpy()

#%% Resampling
# Resample gait sequence from 100 Hz to 40 Hz
accV40 = resamp_interp(gs,fs,algorithm_target_fs)
#%% Filtering
# Get the coefficients of the gait filter (just to perform padding)
b,a = EpflGaitFilter().coefficients
# Padding for short data
len_padd = 10000 * len(b)
accV40_zp = np.pad(accV40, (len_padd, len_padd), 'wrap')
# Filtering: Gait filter + De-drift
filter = EpflDedriftedGaitFilter()
accV40_lpf = filter.filter(accV40_zp,sampling_rate_hz=algorithm_target_fs).filtered_data_
# Remove the padding
accV40_lpf_rmzp = accV40_lpf[len_padd-1:-len_padd]
#%% Cumulative integral
accVLPInt = cumtrapz(accV40_lpf_rmzp,initial = 0) / algorithm_target_fs
#%% Continuous Wavelet Transform (CWT)
optimal_width = 6.4
accVLPIntCwt = cwt(accVLPInt, ricker, [optimal_width])
# Remove the mean from accVLPIntCwt
accVLPIntCwt = accVLPIntCwt - accVLPIntCwt.mean()
#%% Detect ICs
x = accVLPIntCwt.squeeze() # squeeze extra dimensions
# Detect the maximum peaks between the zero crossings
pks1, ipks1 = MaxPeaksBetweenZC(x)
# Filter negative peaks
indx1 = np.where(pks1 < 0)[0].tolist()
IC = np.array([ipks1[i] for i in indx1])
# Convert to seconds.
IC = (IC+1) / algorithm_target_fs


#%% Plot and comparison with Matlab
IC_final = IC + single_trial_with_reference.reference_parameters_['lwb'][0]["Start"]
# IC_matlab = [2.905,3.505,4.105,4.68] # HA, 002, Test5, Trial2
# IC_matlab = [4.48,5.055,5.63,6.18,6.73,7.33,7.93] # HA, 001, Test5, Trial2
IC_matlab = [7.395,7.994999999999999,8.57,9.12,9.645,10.195000000000000,10.745] # MS, 001, Test5, Trial1
# IC_matlab = [4.755,5.33,5.88,6.404999999999999,6.955,7.505,8.055] # MS, 001, Test5, Trial2

IC_matlab_final = [int(x*fs) for x in IC_matlab]
IC_stereo = single_trial_with_reference.reference_parameters_['lwb'][0]["InitialContact_Event"]


plt.close()
fig, ax = plt.subplots()
t = np.arange(1/fs, (len(accV)+1)/fs, 1/fs, dtype=float)
ax.plot(t,accV)
ax.plot(IC_final,accV.array[(IC_final*fs).astype(int)].to_numpy(),'ro',label = 'Python')
ax.plot(IC_matlab,accV.array[IC_matlab_final],'b*',label = 'Matlab')
ax.fill_betweenx(np.arange(0.4, 1.8, 0.01),s/fs,e/fs,facecolor='green', alpha=0.2)
plt.xlabel("Time (s)")
plt.ylabel("Vertical Acceleration (m/s^2)")
plt.title('IC detection: HA002 - Test 5 - Trial 2')
plt.legend(loc="upper left")
plt.show()
