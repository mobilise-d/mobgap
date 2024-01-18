import warnings
from collections import namedtuple
from itertools import groupby
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from gaitmap.utils.array_handling import merge_intervals
from typing_extensions import Unpack, Self

from gaitlink.data_transform import EpflDedriftedGaitFilter
from gaitlink.gsd.base import BaseGsdDetector


def hilbert_envelop(y, Smooth_window, threshold_style, DURATION):
    """NOTE: This has been edited from the original MATLAB version to remove perceived error"""
    # Calculate the analytical signal and get the envelope
    amplitude_envelope = np.abs(scipy.signal.hilbert(y))
    """plt.figure()
    plt.plot(y.T)
    plt.plot(amplitude_envelope.T)"""

    # Take the moving average of analytical signal
    env = np.convolve(
        amplitude_envelope, np.ones(Smooth_window) / Smooth_window, "same"
    )  # Smooth  NOTE: Original matlab code used mode 'full', meaning the length of convolution was different to env, this has been fixed here
    env = env - np.mean(env)  # Get rid of offset
    env = env / np.max(env)  # Normalize

    """ Threshold the signal """
    # Input the threshold if needed
    if not threshold_style:
        f = plt.figure()
        plt.plot(env)
        plt.title("Select a threshold on the graph")
        threshold_sig = input("What threshold have you selected?\n")
        print("You have selected: ", threshold_sig)
        plt.close(f)
    else:
        # Threshold style
        threshold_sig = 4 * np.nanmean(env)
    noise = np.mean(env) * (1 / 3)  # Noise level
    threshold = np.mean(env)  # Signal level

    # Initialize Buffers
    thresh_buff = np.zeros(len(env) - DURATION + 1)
    noise_buff = np.zeros(len(env) - DURATION + 1)
    thr_buff = np.zeros(len(env) + 1)
    h = 1
    alarm = np.zeros(len(env) + 1)

    for i in range(len(thresh_buff)):
        # Update threshold 10% of the maximum peaks found
        if (env[i : i + DURATION] > threshold_sig).all():
            alarm[i] = max(env)
            threshold = 0.1 * np.mean(env[i : i + DURATION])
            h = h + 1
        elif np.mean(env[i : i + DURATION]) < threshold_sig:
            noise = np.mean(env[i : i + DURATION])
        else:
            if noise_buff.any():
                noise = np.mean(noise_buff)

        thresh_buff[i] = threshold
        noise_buff[i] = noise

        # Update threshold
        if h > 1:
            threshold_sig = noise + 0.50 * (abs(threshold - noise))
        thr_buff[i] = threshold_sig

    return [alarm, env]


import numpy as np

def find_min_max(signal: np.ndarray, threshold: float) -> tuple:
    """
    Identify the indices of local minima and maxima in a 1D numpy array (signal),
    where the values are beyond a specified threshold.

    Parameters:
    signal (np.ndarray): A 1D numpy array representing the signal.
    threshold (float): A threshold value to filter the minima and maxima.

    Returns:
    tuple: Two arrays containing the indices of local minima and maxima, respectively.
    """
    signal = signal.squeeze()
    diff = np.diff(signal)
    extrema_indices = np.nonzero(diff[1:] * diff[:-1] <= 0)[0] + 1

    minima = extrema_indices[diff[extrema_indices] >= 0]
    maxima = extrema_indices[diff[extrema_indices] < 0]

    minima = minima[signal[minima] < -threshold]
    maxima = maxima[signal[maxima] > threshold]

    return minima, maxima



def find_pulse_trains(x):
    w = {}
    walkflag = 0
    THD = 3.5 * 40
    n = 0

    start = [0]
    steps = [0]
    end = [0]

    if len(x) > 2:
        for i in range(len(x) - 1):
            if x[i + 1] - x[i] < THD:
                if walkflag == 0:
                    start[n] = x[i]
                    steps[n] = 1
                    walkflag = 1
                else:
                    steps[n] = steps[n] + 1
                    THD = 1.5 * 40 + (x[i] - start[n]) / steps[n]
            else:
                if walkflag == 1:
                    end[n] = x[i - 1]
                    n = n + 1
                    start = start + [0]
                    steps = steps + [0]
                    end = end + [0]
                    walkflag = 0
                    THD = 3.5 * 40

    if walkflag == 1:
        if x[-1] - x[-2] < THD:
            end[-1] = x[-1]
            steps[n] = steps[n] + 1
        else:
            end[-1] = x[-1]

    return np.array([start, end, steps]).T


def Intersect(a, b):
    na = len(a)
    nb = len(b)

    c = np.zeros(shape=(nb, 2))

    if na == 0 or nb == 0:
        warnings.warn("a or b is empty, returning empty c")
        return c

    k = 0
    ia = 0
    ib = 0
    state = 3

    while ia <= na - 1 and ib <= nb - 1:
        if state == 1:
            if a[ia, 1] < b[ib, 0]:
                ia = ia + 1
                state = 3
            elif a[ia, 1] < b[ib, 1]:
                c[k, 0] = b[ib, 0]
                c[k, 1] = a[ia, 1]
                k = k + 1
                ia = ia + 1
                state = 2
            else:
                c[k, :] = b[ib, :]
                k = k + 1
                ib = ib + 1

        elif state == 2:
            if b[ib, 1] < a[ia, 0]:
                ib = ib + 1
                state = 3
            elif b[ib, 1] < a[ia, 1]:
                c[k, 0] = a[ia, 0]
                c[k, 1] = b[ib, 1]
                k = k + 1
                ib = ib + 1
                state = 1
            else:
                c[k, :] = a[ia, :]
                k = k + 1
                ia = ia + 1

        elif state == 3:
            if a[ia, 0] < b[ib, 0]:
                state = 1
            else:
                state = 2

    if (~c.any(axis=1)).any():
        raise ValueError("c has a row of zeros")

    return c


########################################################################################################################
class GsdLowBackAcc(BaseGsdDetector):
    def __init__(self, savgol_order: int = 7):
        # TODO: This still does nothing
        self.savgol_order = savgol_order

    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_: Unpack[dict[str, Any]]) -> Self:
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        gsd_output = gsd_low_back_acc(data[["acc_x", "acc_y", "acc_z"]], sampling_rate_hz, plot_results=True)
        self.gsd_list_ = gsd_output

        return self


def gsd_low_back_acc(acc, fs, plot_results=True):
    """

    :param acc:
    :param fs:
    :param plot_results:
    :return GSD_Output:
    """
    algorithm_target_fs = 40  # Sampling rate required for the algorithm

    # Signal vector magnitude
    accN = np.sqrt(
        np.square(acc.iloc[:, 0].to_numpy())
        + np.square(acc.iloc[:, 1].to_numpy())
        + np.square(acc.iloc[:, 2].to_numpy())
    )

    # Resample to algorithm_target_fs
    accN_resampled = scipy.signal.resample(accN, int(np.round(len(accN) / fs * algorithm_target_fs)))
    # plt.plot(accN_resampled)
    # NOTE: accN_resampled is slightly different in length and values to accN40 in MATLAB, plots look ok though

    # Filter to enhance the acceleration signal, when low SNR, impaired, asymmetric and slow gait
    acc_filtered = scipy.signal.savgol_filter(accN_resampled, polyorder=7, window_length=21)
    acc_filtered = EpflDedriftedGaitFilter().filter(acc_filtered, sampling_rate_hz=40).filtered_data_
    # NOTE: Original MATLAB code calls old version of cwt (open wavelet.internal.cwt in MATLAB to inspect) in
    #   accN_filt3=cwt(accN_filt2,10,'gaus2',1/40);
    #   Here, 10 is the scale, gaus2 is the second derivative of a Gaussian wavelet, aka a Mexican Hat or Ricker wavelet
    #   In Python, a scale of 7 matches the MATLAB scale of 10 from visual inspection of plots (likely due to how to two
    #   languages initialise their wavelets), giving the line below
    acc_filtered = scipy.signal.cwt(acc_filtered.squeeze(), scipy.signal.ricker, [7])
    acc_filtered4 = scipy.signal.savgol_filter(acc_filtered, 11, 5)
    acc_filtered = scipy.signal.cwt(acc_filtered4.squeeze(), scipy.signal.ricker, [7])  # See NOTE above
    acc_filtered = scipy.ndimage.gaussian_filter(
        acc_filtered.squeeze(), 2
    )  # NOTE: sigma = windowWidth / 5, windowWidth = 10 (from MATLAB)
    acc_filtered = scipy.ndimage.gaussian_filter(
        acc_filtered.squeeze(), 2
    )  # NOTE: sigma = windowWidth / 5, windowWidth = 10 (from MATLAB)
    acc_filtered = scipy.ndimage.gaussian_filter(
        acc_filtered.squeeze(), 3
    )  # NOTE: sigma = windowWidth / 5, windowWidth = 15 (from MATLAB)
    acc_filtered = scipy.ndimage.gaussian_filter(
        acc_filtered.squeeze(), 2
    )  # NOTE: sigma = windowWidth / 5, windowWidth = 10 (from MATLAB)
    """plt.figure()
    plt.plot(acc_filtered.T)"""

    sigDetActv = acc_filtered

    # Find pre-detection of 'active' periods in order to estimate the amplitude of acceleration peaks
    [alarm, _] = hilbert_envelop(
        sigDetActv, algorithm_target_fs, True, algorithm_target_fs
    )  # NOTE: This has been edited from the original MATLAB version to remove perceived error
    walkLowBack = np.array([])

    if alarm.any():  # If any alarms detected
        len_alarm = [
            len(list(s)) for v, s in groupby(alarm, key=lambda x: x > 0)
        ]  # Length of each consecutive stretch of nonzero values in alarm
        end_alarm = np.cumsum(len_alarm)
        start_alarm = np.concatenate([np.array([0]), end_alarm[:-1]])
        alarmed = [
            v for v, s in groupby(alarm, key=lambda x: x > 0)
        ]  # Whether each consecutive stretch of nonzero values in alarm is alarmed

        for s, e, a in zip(start_alarm, end_alarm, alarmed):  # Iterate through the consecutive periods
            if a:  # If alarmed
                if e - s <= 3 * algorithm_target_fs:  # If the length of the alarm period is too short
                    alarm[s:e] = 0  # Replace this section of alarm with zeros
                else:
                    walkLowBack = np.concatenate([walkLowBack, sigDetActv[s - 1 : e - 1]])

        if walkLowBack.size != 0:
            peaks_p, _ = scipy.signal.find_peaks(walkLowBack)
            peaks_n, _ = scipy.signal.find_peaks(-walkLowBack)
            pksp, pksn = walkLowBack[peaks_p], -walkLowBack[peaks_n]
            pks = np.concatenate([pksp[pksp > 0], pksn[pksn > 0]])
            th = np.percentile(pks, 5)  # data adaptive threshold
            f = sigDetActv

        else:  # If hilbert_envelope fails to detect 'active' try version [1]
            th = 0.15
            f = acc_filtered4

    else:  # If hilbert_envelope fails to detect 'active' try version [1]
        th = 0.15
        f = acc_filtered4

    # mid - swing detection
    min_peaks, max_peaks= find_min_max(f, th)

    MIN_N_STEPS = 5

    gs_from_max = find_pulse_trains(max_peaks)
    gs_from_min = find_pulse_trains(min_peaks)
    gs_from_max = gs_from_max[gs_from_max[:, 2] >= MIN_N_STEPS]
    gs_from_min = gs_from_min[gs_from_min[:, 2] >= MIN_N_STEPS]

    combined_final = Intersect(gs_from_max[:, :2], gs_from_min[:, :2])

    if combined_final.size == 0:
        return pd.DataFrame(columns=["start", "end"]).astype(int)

    # Find all max_peaks withing each final gs
    steps_per_gs = [[x for x in max_peaks if gs[0] <= x <= gs[1]] for gs in combined_final]
    n_steps_per_gs = np.array([len(steps) for steps in steps_per_gs])
    mean_step_times = np.array([np.mean(np.diff(steps)) for steps in steps_per_gs])

    # Pad each gs by 0.75*step_time before and after
    combined_final[:, 0] = np.fix(combined_final[:, 0] - 0.75 * mean_step_times)
    combined_final[:, 1] = np.fix(combined_final[:, 1] + 0.75 * mean_step_times)

    # Filter again by Number steps
    combined_final = combined_final[n_steps_per_gs >= MIN_N_STEPS]

    if combined_final.size == 0:
        return pd.DataFrame(columns=["start", "end"]).astype(int)

    MAX_GAP_S = 3
    combined_final = merge_intervals(combined_final, algorithm_target_fs * MAX_GAP_S)

    # Convert back to original sampling rate
    combined_final = combined_final * fs / algorithm_target_fs

    # Cap the start and the end of the signal using clip
    combined_final = np.clip(combined_final, 0, len(acc))

    return pd.DataFrame(combined_final, columns=["start", "end"]).astype(int)
