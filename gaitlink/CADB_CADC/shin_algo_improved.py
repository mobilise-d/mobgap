import os

import numpy as np
from gaitmap.data_transform import Resample

from gaitlink.CADB_CADC.zerocros import zerocros
from scipy.signal import savgol_filter, cwt, ricker
from scipy.ndimage import gaussian_filter
import csv
from gaitlink.data_transform import EpflDedriftedGaitFilter

from resampInterp import resampInterp


def shin_algo_improved(imu_acc, fs, data='norm'):

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

    csv_file = "epfl_gait_filter.csv"
    Num = []  # Initialize the list to store coefficients

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row if there is one
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
    accN_filt4 = savgol_filter(accN_filt3.squeeze(), window_length=11, polyorder=5)
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

    # Resample to 50Hz for output in seconds or to 100 for IC detection
    # Resample to 50Hz for consistency with the original paper
    #fs_new = 50
    #accN_MultiFilt_rmp50 = resampInterp(accN_MultiFilt_rmp, algorithm_target_fs, fs_new);

    # Initial contacts timings (heel strike events) detected as positive slopes zero-crossing in seconds
    #IC_lowSNR = zerocros(accN_MultiFilt_rmp50, 'p')
    #IC_lowSNR = IC_lowSNR[0]
    #IC_lowSNR = np.round(IC_lowSNR)
    #IC = IC_lowSNR / fs_new # in seconds

    # Resample to 100Hz for consistency with the original data (for ICD) or to 50 for consistency with original paper
    current_sampling_rate = 40
    target_sampling_rate = 100
    resampler = Resample(target_sampling_rate)
    resampler.transform(data=accN_MultiFilt_rmp, sampling_rate_hz=current_sampling_rate)
    accN_MultiFilt_rmp100 = resampler.transformed_data_

    # Initial contacts timings (heel strike events) detected as positive slopes zero-crossing in sample 120
    IC_lowSNR = zerocros(accN_MultiFilt_rmp100, 'p')
    IC_lowSNR = IC_lowSNR[0]
    IC_lowSNR = np.round(IC_lowSNR)
    IC = IC_lowSNR  # in sample
   #IC = IC / target_sampling_rate  # in seconds

    return IC