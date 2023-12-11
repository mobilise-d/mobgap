import numpy as np
from gaitmap.data_transform import Resample
from scipy.signal import savgol_filter, cwt, ricker
from scipy.ndimage import label, binary_closing, binary_opening, gaussian_filter, grey_closing, grey_opening
from gaitmap.utils.array_handling import bool_array_to_start_end_array
import csv

from gaitlink.data_transform import EpflDedriftedGaitFilter
from resampInterp import resampInterp


def hklee_algo_improved(imu_acc, fs, data='norm'):

    # Check if 'data' is provided, if not, set it to the default 'norm' (all axes of accelerometry)
    if not data or len(data) == 0:
        data = 'norm'

    acc = imu_acc

    if 'x' in data:
        accN = acc.flatten()

    elif 'norm' in data:
        accN = np.sqrt(acc[:, 0] ** 2 + acc[:, 1] ** 2 + acc[:, 2] ** 2)

    IC = []
    IC_lowSNR = []


    # Resample to 40Hz to process with filters
    current_sampling_rate = fs
    target_sampling_rate = 40
    resampler = Resample(target_sampling_rate)
    resampler.transform(data=accN, sampling_rate_hz=current_sampling_rate)
    accN40 = resampler.transformed_data_

    # FIR filter TODO: update coefficients from import function
    Num = []
    csv_file = "epfl_gait_filter.csv"

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            coefficient = float(row[0])
            Num.append(coefficient)

    # Padding to cope with short data
    len_padd = 4 * len(Num)
    accN40_zp = np.pad(accN40, (len_padd, len_padd), 'wrap')


    #TODO: change number on files below
    # Filters
    # 1
    accN_filt1 = savgol_filter(accN40_zp.squeeze(), window_length=21, polyorder=7)
    # 2
    filter = EpflDedriftedGaitFilter()
    accN_filt2 = filter.filter(accN_filt1, sampling_rate_hz = 40).filtered_data_
    # 3
    accN_filt3 = cwt(accN_filt1.squeeze(), ricker, [10])
    # 4
    accN_filt4 = savgol_filter(accN_filt1.squeeze(), window_length=11, polyorder=5)
    # 5
    accN_filt5 = cwt(accN_filt4.squeeze(), ricker, [10])
    # 6
    windowWidth = 10
    sigma = windowWidth / 5
    accN_filt6 = gaussian_filter(accN_filt5.squeeze(), sigma)
    # 7
    windowWidth = 10
    sigma = windowWidth / 5
    accN_filt7 = gaussian_filter(accN_filt6.squeeze(), sigma)
    # 8
    windowWidth = 15
    sigma = windowWidth / 5
    accN_filt8 = gaussian_filter(accN_filt7.squeeze(), sigma)
    # MultiFilt
    accN_MultiFilt_rmp = accN_filt8[len_padd:-len_padd]


    # Resample to 100Hz for consistency with the original data (for ICD) or to 120 for consistency with original paper
    current_sampling_rate = 40
    target_sampling_rate = 100
    resampler = Resample(target_sampling_rate)
    resampler.transform(data=accN_MultiFilt_rmp, sampling_rate_hz=current_sampling_rate)
    accN_MultiFilt_rmp120 = resampler.transformed_data_


    #fs_new = 100
    #accN_MultiFilt_rmp120 = resampInterp(accN_MultiFilt_rmp, algorithm_target_fs, fs_new)

    # Apply morphological filters
    SE_closing = np.ones(32, dtype=int)
    SE_opening = np.ones(18, dtype=int)

    C = grey_closing(accN_MultiFilt_rmp120, structure=SE_closing)
    O = grey_opening(C, structure=SE_opening)
    R = C - O

    if np.any(R > 0):
        idx = bool_array_to_start_end_array(R > 0)
        IC_lowSNR = np.zeros(len(idx), dtype=int)
        for j in range(len(idx)):
            start_idx, end_idx = idx[j, 0], idx[j, 1]
            values_within_range = R[start_idx:end_idx + 1]
            imax = start_idx + np.argmax(values_within_range)

            # Assign the value to the NumPy array
            IC_lowSNR[j] = imax


        # IC in seconds
    IC = IC_lowSNR
    #IC = IC_lowSNR / target_sampling_rate

    return IC


# Plot to verify accuracy of step detection (IC in samples)
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(np.arange(len(accN_MultiFilt_rmp120)), accN_MultiFilt_rmp120, 'k')
    # plt.plot(IC_lowSNR, accN_MultiFilt_rmp120[IC_lowSNR], 'ro', linewidth=1.4)
    # plt.legend(['accNorm (pre-processed)', 'Initial Contacts (IC)'])
    # plt.subplot(212)
    # plt.plot(R, 'r')
    # plt.legend(['accNorm (pre-processed and applied morphological filters)'])
    # plt.show()