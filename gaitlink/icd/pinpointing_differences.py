from gaitlink.data import LabExampleDataset
from gaitlink.data_transform._filter import EpflDedriftedGaitFilter
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import filtfilt, cwt, ricker
from scipy.io import loadmat
from scipy.integrate import cumtrapz
import numpy as np

example_data = LabExampleDataset()
# %%
# You can select the data you want using the ``get_subset`` method.
ha_example_data = example_data.get_subset(cohort="HA")
ha_example_data

# %%
# Once you selected only a single row of the dataset (either by repeated ``get_subset`` or by iteration), you can load
# the actual data.
single_test = ha_example_data.get_subset(participant_id="002", test="Test5", trial="Trial2")
single_test

# %%
# The raw IMU data:
imu_data = single_test.data["LowerBack"]
# vertical acceleration
accV = imu_data["acc_x"]/9.81
# accV = imu_data["acc_x"]

# %%
# You can also load the reference system data, by specifying the ``reference_system`` argument.
# All parameters related to the reference systems have a trailing underscore.
example_data_with_reference = LabExampleDataset(reference_system="Stereophoto")
single_trial_with_reference = example_data_with_reference.get_subset(
    cohort="HA", participant_id="002", test="Test5", trial="Trial2"
)
fs = single_trial_with_reference.sampling_rate_hz
# -1 with respect to Matlab (Python starts counting from 0 )
s = round(single_trial_with_reference.reference_parameters_['lwb'][0]["Start"]*fs)-1
e = round(single_trial_with_reference.reference_parameters_['lwb'][0]["End"]*fs)-1
gs = accV.array[s:e+1].to_numpy()




def resampInterp(y, fs_initial, fs_final):
    recordingTime = len(y)
    x = np.arange(1, recordingTime + 1)  # MATLAB uses 1-based indexing
    xq = np.arange(1, recordingTime + 1, fs_initial / fs_final)  # Create the new time vector
    interp_func = interp1d(x, y, kind='linear', axis=0, fill_value='extrapolate') # returns an interpolation function
    yResamp = interp_func(xq)
    return yResamp

def RemoveDrift40Hz(x):
#  s = RemoveDrift40Hz(x)
# Removes gyro's drift using an IIR filter
    s = filtfilt([1, -1], [1, -.9748], x)
    return s

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



# resampling frequency
algorithm_target_fs=40
# resampling
accV40 = resampInterp(gs,fs,algorithm_target_fs)

# Load the filter coefficients
cwt_matlab = loadmat('C:\\Users\\paolo\\OneDrive - Politecnico di Torino\\Borsa Polito\\Progetti\\Rewrite MobiliseD\\Mobilise-D-TVS-Recommended-Algorithms\\ICDA\\cwt_matlab_HA02_Test5_Trial2.mat')
cwt_matlab = cwt_matlab["accVLPIntCwt"]
int_matlab = loadmat('C:\\Users\\paolo\\OneDrive - Politecnico di Torino\\Borsa Polito\\Progetti\\Rewrite MobiliseD\\Mobilise-D-TVS-Recommended-Algorithms\\ICDA\\int_matlab_HA02_Test5_Trial2.mat')
int_matlab = int_matlab["accVLPInt"]
Num = loadmat('C:\\Users\\paolo\\OneDrive - Politecnico di Torino\\Borsa Polito\\Progetti\\Rewrite MobiliseD\\gaitlink\\gaitlink\\icd\\FIR-2-3Hz-40.mat')
Num = Num['Num'][0]
imu_acc_matlab = loadmat('C:\\Users\\paolo\\OneDrive - Politecnico di Torino\\Borsa Polito\\Progetti\\Rewrite MobiliseD\\Mobilise-D-TVS-Recommended-Algorithms\\ICDA\\imu_acc_matlab_HA02_Test5_Trial2.mat')
imu_acc_matlab = imu_acc_matlab["imu_acc"]
acc_V_40_matlab = loadmat('C:\\Users\\paolo\\OneDrive - Politecnico di Torino\\Borsa Polito\\Progetti\\Rewrite MobiliseD\\Mobilise-D-TVS-Recommended-Algorithms\\ICDA\\acc_V_40_matlab_HA02_Test5_Trial2.mat')
acc_V_40_matlab = acc_V_40_matlab["accV40"]
accV40_lpf_rmzp_matlab = loadmat('C:\\Users\\paolo\\OneDrive - Politecnico di Torino\\Borsa Polito\\Progetti\\Rewrite MobiliseD\\Mobilise-D-TVS-Recommended-Algorithms\\ICDA\\accV40_lpf_rmzp_matlab_HA02_Test5_Trial2.mat')
accV40_lpf_rmzp_matlab = accV40_lpf_rmzp_matlab["accV40_lpf_rmzp"]
# Pinpointing differences - Raw acceleration: full trial of accV
# plt.close()
# fig, ax = plt.subplots()

# t = np.arange(1/100, (len(accV)+1)/100, 1/100, dtype=float)
# ax.plot(t,accV,label='python')
# ax.plot(t,imu_acc_matlab[:,0],label='matlab')
# plt.xlabel("Time (s)")
# plt.ylabel("Vertical Acceleration (m/s^2)")
# plt.title('IC detection: HA002 - Test 5 - Trial 2')
# plt.legend(loc="upper left")
# plt.show()
# differenze = accV-imu_acc_matlab[:,0]
# differenze.mean()

# NO SIGNIFICANT DIFFERENCES--> OK

## Pinpointing differences - Resampling: full trial of accV
# plt.close()
# fig, ax = plt.subplots()
# t = np.arange(1/algorithm_target_fs, (len(accV40)+1)/algorithm_target_fs, 1/algorithm_target_fs, dtype=float)
# ax.plot(t,accV40,label='python')
# ax.plot(t,acc_V_40_matlab,label='matlab')
# plt.xlabel("Time (s)")
# plt.ylabel("Vertical Acceleration (m/s^2)")
# plt.title('IC detection: HA002 - Test 5 - Trial 2')
# plt.legend(loc="upper left")
# plt.show()
# differenze = accV40-acc_V_40_matlab
# differenze.mean()

# Padding for short data
len_padd = 10000 * len(Num)
accV40_zp = np.pad(accV40, (len_padd, len_padd), 'wrap')

# # Plot zero padding
# plt.close()
# fig, ax = plt.subplots()
# t = np.arange(1/algorithm_target_fs, (len(accV40_zp)+1)/algorithm_target_fs, 1/algorithm_target_fs, dtype=float)
# ax.plot(t,accV40_zp,label='python')
# plt.show()

# Apply the filter with filtfilt
filter = EpflDedriftedGaitFilter()
accV40_lpf = filter.filter(accV40_zp,sampling_rate_hz=algorithm_target_fs).filtered_data_
# de_drifted_accV40_zp = RemoveDrift40Hz(accV40_zp)
# accV40_lpf = filtfilt(Num, 1, de_drifted_accV40_zp)
# Remove the padding
accV40_lpf_rmzp = accV40_lpf[len_padd-1:-len_padd] # +1:

# Pinpointing differences - filtering
# plt.close()
# fig, ax = plt.subplots()
# t = np.arange(1/100, (len(accV40_lpf_rmzp)+1)/100, 1/100, dtype=float)
# ax.plot(t,accV40_lpf_rmzp,label='python')
# ax.plot(t,accV40_lpf_rmzp_matlab,label='matlab')
# plt.xlabel("Time (s)")
# plt.ylabel("Vertical Acceleration (m/s^2)")
# plt.title('IC detection: HA002 - Test 5 - Trial 2')
# plt.legend(loc="upper left")
# plt.show()
# differenze = accV40_lpf_rmzp-accV40_lpf_rmzp_matlab
# differenze.mean()

# Calculate cumulative integral (cumtrapz) of accV40_lpf_rmzp
accVLPInt = cumtrapz(accV40_lpf_rmzp,initial = 0) / algorithm_target_fs

# plt.close()
# fig, ax = plt.subplots()
# t = np.arange(1/algorithm_target_fs, (len(accVLPInt)+1)/algorithm_target_fs, 1/algorithm_target_fs, dtype=float)
# ax.plot(t,accVLPInt,label='python')
# ax.plot(t,int_matlab[:,0],label='matlab')
# plt.xlabel("Time (s)")
# plt.ylabel("Vertical Acceleration (m/s^2)")
# plt.title('IC detection: HA002 - Test 5 - Trial 2')
# plt.legend(loc="upper left")
# plt.show()
# differenze = accVLPInt-int_matlab[:,0]
# differenze.mean()
# IC
# Continuous Wavelet Transform (CWT)
# for loop to optimize the width parameter of cwt-> optimal width: 6.4
# minimum = 10000
# optimal_width = 0
# errori = np.zeros([1,100])
# for i in np.arange(1,101):
#     accVLPIntCwt = cwt(accVLPInt, ricker, [i/10])
#     accVLPIntCwt = accVLPIntCwt - accVLPIntCwt.mean()
#     differenze = accVLPIntCwt[0, :] - cwt_matlab[0, :]
#     errori[0,i-1] = sum(abs(differenze))
#     print(i)
#     print(sum(abs(differenze)))
#     if sum(abs(differenze)) < minimum:
#         minimum = sum(abs(differenze))
#         optimal_width = i/10
# np.argmin(errori)
# plt.close()
# fig, ax = plt.subplots()
# t = np.arange(1, 101, dtype=float)
# ax.plot(t,errori[0,:],label='python')
# plt.show()
optimal_width = 6.4
accVLPIntCwt = cwt(accVLPInt, ricker, [optimal_width])
# Remove the mean from accVLPIntCwt
accVLPIntCwt = accVLPIntCwt - accVLPIntCwt.mean()

# plt.close()
# fig, ax = plt.subplots()
# t = np.arange(1/algorithm_target_fs, (len(accVLPInt)+1)/algorithm_target_fs, 1/algorithm_target_fs, dtype=float)
# ax.plot(t,accVLPIntCwt[0,:],label='python')
# ax.plot(t,cwt_matlab[0,:],label='matlab')
# plt.xlabel("Time (s)")
# plt.ylabel("Vertical Acceleration (m/s^2)")
# plt.title('IC detection: HA002 - Test 5 - Trial 2')
# plt.legend(loc="upper left")
# plt.show()
# differenze = accVLPIntCwt[0,:]-cwt_matlab[0,:]
# print(differenze.mean())

x = accVLPIntCwt.squeeze()
pks1, ipks1 = MaxPeaksBetweenZC(x)
# pks1, ipks1 = MaxPeaksBetweenZC(accVLPIntCwt)



# Filter negative peaks
indx1 = np.where(pks1 < 0)[0].tolist()
IC = np.array([ipks1[i] for i in indx1])
IC = (IC+1) / algorithm_target_fs  # Convert to seconds

# FC
# Continuous Wavelet Transform (CWT)
accVLPIntCwt2 = cwt(accVLPIntCwt.squeeze(), ricker, [6])

# accVLPIntCwt2, _ = pywt.cwt(accVLPIntCwt, scales=[9], wavelet='gaus2', sampling_period=1/algorithm_target_fs)

# Remove the mean from accVLPIntCwt
accVLPIntCwt2 = accVLPIntCwt2 - accVLPIntCwt2.mean()

# Find peaks and indices
x2 = accVLPIntCwt2.squeeze()
pks2, ipks2 = MaxPeaksBetweenZC(x2)

# Filter positive peaks
indx2 = np.where(pks2 > 0)[0].tolist()
FC = np.array([ipks2[i] for i in indx2])
FC = FC / algorithm_target_fs  # Convert to seconds

IC_final = IC + single_trial_with_reference.reference_parameters_['lwb'][0]["Start"]
FC_final = FC + single_trial_with_reference.reference_parameters_['lwb'][0]["Start"]


IC_matlab = [2.905000000000000,3.505000000000000,4.10500000000000,4.68000000000000]
IC_matlab_final = [int(x*100) for x in IC_matlab]
IC_stereo = single_trial_with_reference.reference_parameters_['lwb'][0]["InitialContact_Event"]


plt.close()
fig, ax = plt.subplots()

t = np.arange(1/100, (len(accV)+1)/100, 1/100, dtype=float)
ax.plot(t,accV)
ax.plot(IC_final,accV.array[(IC_final*100).astype(int)].to_numpy(),'go',label = 'Python')
ax.plot(IC_matlab,accV.array[IC_matlab_final],'b*',label = 'Matlab')

ax.fill_betweenx(np.arange(0.4, 1.8, 0.01),s/100,e/100,facecolor='green', alpha=0.2)
# ax.vlines(x = IC_final,ymin = 0.4, ymax = 1.8, colors = 'b', label = 'Python')
# ax.vlines(x = IC_matlab,ymin = 0.4, ymax = 1.8, colors = 'r', label = 'Matlab')
# ax.vlines(x = IC_stereo,ymin = 0.4, ymax = 1.8, colors = 'g', label = 'Stereophoto')

plt.xlabel("Time (s)")
plt.ylabel("Vertical Acceleration (m/s^2)")
plt.title('IC detection: HA002 - Test 5 - Trial 2')
plt.legend(loc="upper left")
plt.show()
