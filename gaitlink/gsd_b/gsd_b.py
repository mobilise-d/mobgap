import numpy as np
import scipy
import matplotlib.pyplot as plt
import warnings
from gaitlink.data_transform import EpflDedriftedGaitFilter
from itertools import groupby


def hilbert_envelop(y, Smooth_window, threshold_style, DURATION):

    """NOTE: This has been edited from the original MATLAB version to remove perceived error """

    # Calculate the analytical signal and get the envelope
    amplitude_envelope = np.abs(scipy.signal.hilbert(y))
    """plt.figure()
    plt.plot(y.T)
    plt.plot(amplitude_envelope.T)"""

    # Take the moving average of analytical signal
    env = np.convolve(amplitude_envelope, np.ones(Smooth_window)/Smooth_window, 'same')  # Smooth  NOTE: Original matlab code used mode 'full', meaning the length of convolution was different to env, this has been fixed here
    env = env - np.mean(env)  # Get rid of offset
    env = env / np.max(env)  # Normalize

    """ Threshold the signal """
    # Input the threshold if needed
    if not threshold_style:
        f = plt.figure()
        plt.plot(env)
        plt.title('Select a threshold on the graph')
        threshold_sig = input('What threshold have you selected?\n')
        print('You have selected: ', threshold_sig)
        plt.close(f)
    else:
        # Threshold style
        threshold_sig = 4*np.nanmean(env)
    noise = np.mean(env)*(1/3)  # Noise level
    threshold = np.mean(env)  # Signal level

    # Initialize Buffers
    thresh_buff = np.zeros(len(env) - DURATION + 1)
    noise_buff = np.zeros(len(env) - DURATION + 1)
    thr_buff = np.zeros(len(env)+1)
    h = 1
    alarm = np.zeros(len(env)+1)

    for i in range(len(thresh_buff)):

        # Update threshold 10% of the maximum peaks found
        if (env[i: i + DURATION] > threshold_sig).all():
            alarm[i] = max(env)
            threshold = 0.1 * np.mean(env[i:i + DURATION])
            h = h + 1
        else:

            # Update noise
            if np.mean(env[i: i + DURATION]) < threshold_sig:
                noise = np.mean(env[i:i + DURATION])
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


def FindMinMax(s, thr):
    s = s.squeeze()
    d = np.diff(s)
    f = np.nonzero(d[1:] * d[:-1] <= 0)[0]
    f = f + 1

    mi = f[d[f] >= 0]
    ma = f[d[f] < 0]

    ma = ma[s[ma] > thr]
    mi = mi[s[mi] < -thr]

    return mi, ma


def FindPulseTrains(x):

    w = {}
    walkflag = 0
    THD = 3.5 * 40
    n = 0

    start = [0]
    steps = [0]
    end = [0]

    if len(x) > 2:
        for i in range(len(x)-1):
            if x[i+1] - x[i] < THD:
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

    return {'start': np.array(start), 'steps': np.array(steps), 'end': np.array(end)}


def ConvertWtoSet(w):
    s = np.zeros((len(w['start']), 2))
    s[:, 0] = w['start']
    s[:, 1] = w['end']
    return s


def Intersect(a, b):
    na = len(a)
    nb = len(b)

    c = np.zeros(shape=(nb, 2))

    if na == 0 or nb == 0:
        warnings.warn('a or b is empty, returning empty c')
        return c

    k = 0
    ia = 0
    ib = 0
    state = 3

    while ia <= na-1 and ib <= nb-1:
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
        raise ValueError('c has a row of zeros')

    return c


def pack_results(periods, peaks):
    n = len(periods)
    w = {'start': periods[:, 0], 'end': periods[:, 1], 'steps': np.zeros(n), 'mid_swing': []}
    mid_swing = []

    for i in range(n):
        steps = peaks[np.logical_and(peaks >= w['start'][i], peaks <= w['end'][i])]
        w['steps'][i] = len(steps)
        w['mid_swing'].append(steps)
        mid_swing.append(steps)

        # Increase the duration of reported walking periods by half a step_time before the first and after the last step
        if len(steps) > 2:
            step_time = np.mean(np.diff(steps))
            w['start'][i] = np.fix(w['start'][i] - 1.5 * step_time / 2)
            w['end'][i] = np.fix(w['end'][i] + 1.5 * step_time / 2)

    mid_swing = np.concatenate(mid_swing).sort()

    # check to see if any two consecutive detected walking periods are overlapped; if yes, join them. (This should not normally happen though!)
    i = 0
    while i < n-1:
        if w['end'][i] >= w['start'][i+1]:
            w['end'][i] = w['end'][i + 1]
            w['steps'][i] = w['steps'][i] + w['steps'][i + 1]
            w['mid_swing'][i] = ['mid_swing'][i] + w['mid_swing'][i+1]
            w['start'][i + 1] = []
            w['end'][i + 1] = []
            w['steps'][i + 1] = []
            w['mid_swing'][i + 1] = []
            n = n - 1
        else:
            i = i + 1

    return w, mid_swing


########################################################################################################################

def GSD_LowBackAcc(acc, fs, plot_results=True):
    """

    :param acc:
    :param fs:
    :param plot_results:
    :return GSD_Output:
    """

    GSD_Output = {'Start': [], 'End': [], 'fs': []}
    algorithm_target_fs = 40  # Sampling rate required for the algorithm

    # Signal vector magnitude
    accN = np.sqrt(np.square(acc.iloc[:, 0].to_numpy()) + np.square(acc.iloc[:, 1].to_numpy()) + np.square(acc.iloc[:, 2].to_numpy()))

    # Resample to algorithm_target_fs
    accN_resampled = scipy.signal.resample(accN, int(np.round(len(accN)/fs*algorithm_target_fs)))
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
    acc_filtered = scipy.ndimage.gaussian_filter(acc_filtered.squeeze(), 2)  # NOTE: sigma = windowWidth / 5, windowWidth = 10 (from MATLAB)
    acc_filtered = scipy.ndimage.gaussian_filter(acc_filtered.squeeze(), 2)  # NOTE: sigma = windowWidth / 5, windowWidth = 10 (from MATLAB)
    acc_filtered = scipy.ndimage.gaussian_filter(acc_filtered.squeeze(), 3)  # NOTE: sigma = windowWidth / 5, windowWidth = 15 (from MATLAB)
    acc_filtered = scipy.ndimage.gaussian_filter(acc_filtered.squeeze(), 2)  # NOTE: sigma = windowWidth / 5, windowWidth = 10 (from MATLAB)
    """plt.figure()
    plt.plot(acc_filtered.T)"""

    sigDetActv = acc_filtered

    # Find pre-detection of 'active' periods in order to estimate the amplitude of acceleration peaks
    [alarm, _] = hilbert_envelop(sigDetActv, algorithm_target_fs, True, algorithm_target_fs)  # NOTE: This has been edited from the original MATLAB version to remove perceived error
    walkLowBack = np.array([])

    if alarm.any():  # If any alarms detected
        len_alarm = [len(list(s)) for v, s in groupby(alarm, key=lambda x: x > 0)]  # Length of each consecutive stretch of nonzero values in alarm
        end_alarm = np.cumsum(len_alarm)
        start_alarm = np.concatenate([np.array([0]), end_alarm[:-1]])
        alarmed = [v for v, s in groupby(alarm, key=lambda x: x > 0)]  # Whether each consecutive stretch of nonzero values in alarm is alarmed

        for s, e, a in zip(start_alarm, end_alarm, alarmed):  # Iterate through the consecutive periods
            if a:  # If alarmed
                if e-s <= 3 * algorithm_target_fs:  # If the length of the alarm period is too short
                    alarm[s: e] = 0  # Replace this section of alarm with zeros
                else:
                    walkLowBack = np.concatenate([walkLowBack, sigDetActv[s-1:e-1]])

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
    [min_peaks, max_peaks] = FindMinMax(f, th)

    t1 = FindPulseTrains(max_peaks)
    # Only keep walking bouts longer than 4 steps
    t1['start'] = np.array([start for step, start in zip(t1['steps'], t1['start']) if step > 4])
    t1['end'] = np.array([start for step, start in zip(t1['steps'], t1['end']) if step > 4])
    t1['steps'] = np.array([x for x in t1['steps'] if x > 4])

    t2 = FindPulseTrains(min_peaks)
    # Only keep walking bouts longer than 4 steps
    t2['start'] = np.array([start for step, start in zip(t2['steps'], t2['start']) if step > 4])
    t2['end'] = np.array([start for step, start in zip(t2['steps'], t2['end']) if step > 4])
    t2['steps'] = np.array([x for x in t2['steps'] if x > 4])

    t_final = Intersect(ConvertWtoSet(t1), ConvertWtoSet(t2))

    if t_final.any():
        [w, mid_swing] = pack_results(t_final, max_peaks)

        if not w['mid_swing']:
            w['start'][0] = np.max([1, w['start'][0]])
            w['end'][-1] = np.min([w['end'][-1], len(sigDetActv)])
    else:
        w = {}
        MidSwing = []

    n = np.max([len(x) for x in w.values()])
    w_new = {'start': [], 'end': []}
    for j in range(n):
        if w['steps'][j] >= 5:
            w_new['start'].append(w['start'][j])
            w_new['end'].append(w['end'][j])

    walkLabel = np.zeros((len(sigDetActv)))
    n = np.max([len(x) for x in w_new.values()])
    for j in range(n):
        walkLabel[int(w_new['start'][j]): int(w_new['end'][j])] = np.ones(int(w_new['end'][j]-w_new['start'][j]))

    # Merge walking bouts if break less than 3 seconds
    len_noWk = [len(list(s)) for v, s in groupby(walkLabel, key=lambda x: x == 0)]  # Length of each consecutive stretch of nonzero values in alarm
    end_noWk = np.cumsum(len_noWk)
    start_noWk = np.concatenate([np.array([0]), end_noWk[:-1]])
    noWk = [v for v, s in groupby(walkLabel, key=lambda x: x == 0)]  # Whether each consecutive stretch of nonzero values in alarm is alarmed
    start_noWk = start_noWk[noWk]
    end_noWk = end_noWk[noWk]
    if not np.size(start_noWk) == 0:  # If there are periods of non-walking
        for start, end in zip(start_noWk, end_noWk):
            if end - start <= algorithm_target_fs * 3:
                walkLabel[start: end] = np.ones(end-start)  # Merge if non-walking period is too short

    walk = {'start': [], 'end': []}
    if not np.size(np.where(walkLabel == 1)) == 0:
        len_walk = [len(list(s)) for v, s in groupby(walkLabel, key=lambda x: x == 1)]  # Length of each consecutive stretch of nonzero values in alarm
        end_walk = np.cumsum(len_walk)
        start_walk = np.concatenate([np.array([0]), end_walk[:-1]])
        if_walk = [v for v, s in groupby(walkLabel, key=lambda x: x == 1)]  # Whether each consecutive stretch of nonzero values in alarm is alarmed
        start_walk = start_walk[if_walk]
        end_walk = end_walk[if_walk]
        if not np.size(start_walk) == 0:
            for start, end in zip(start_walk, end_walk):
                walk['start'].append(start)
                walk['end'].append(end)

        n = np.max(len(walk['start']))
        for j in range(n):
            GSD_Output['Start'].append((walk['start'][j]) * ( fs / algorithm_target_fs))
            GSD_Output['End'].append((walk['end'][j]) * ( fs / algorithm_target_fs))
            GSD_Output['fs'].append(fs)
    else:
        print('No gait sequence(s) detected')

    # Plot if plot_results==True
    if plot_results:
        plt.figure()
        plt.plot(sigDetActv, label='Filtered accNorm')
        plt.plot(walkLabel*9.81, label='Walking')
        plt.legend()
        plt.xlabel('Samples (40Hz)')

    return GSD_Output