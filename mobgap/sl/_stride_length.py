import numpy as np #TODO: check that all packages are actually needed
from hampel import hampel
import pandas as pd
from scipy.signal import butter, lfilter
from typing import Any
from typing_extensions import Self, Unpack

from gaitmap.trajectory_reconstruction.orientation_methods import MadgwickAHRS
from gaitmap.utils.rotations import rotate_dataset_series
from mobgap.sl.base import BaseSlCalculator, base_sl_docfiller

@base_sl_docfiller
class SlZijlstra(BaseSlCalculator):
    """Implementation of the sl algorithm by Zijlstra and Hof (2003) [1]_ modified by Soltani et al. (2021) [2]_.

    The algorithm includes the following steps starting from vertical acceleration
    of the lower-back during a gait sequence:

    1. Sensor alignment (optional): Madgwick complementary filter
    2. High-pass filtering --> lower cut-off: 0.1 Hz, filter design: Butterworth IIR, filter order: 4
    3. Integration of vertical acceleration --> vertical speed
    4. Drift removal (high-pass filtering) --> lower cut-off: 1 Hz, filter design: Butterworth IIR, filter order: 4
    5. Integration of vertical speed --> vertical displacement d(t)
    6. Compute total vertical displacement during the step (d_step):
        d_step = |max(d(t)) - min(d(t))|
    7. Biomechanical model:
        StepLength = A * 2 * sqrt(2 * LBh * d_step - d_step^2)
        A: tuning coefficient, optimized by training for each population
        LBh: sensor height in meters, representative of the height of the center of mass
    8. Remove step length outliers during the gait sequence --> Hampel filter based on median absolute deviation
    9. Interpolation of StepLength values --> Length per second values
    10. Remove length per second outliers during the gait sequence --> Hampel filter based on median absolute deviation


    This is based on the implementation published as part of the mobilised project [3]_.
    However, this implementation deviates from the original implementation in some places.
    For details, see the notes section and the examples.

    Parameters
    ----------

    beta
        The beta value for the (optional) reorientation with Madgwick filter.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(sl_list_)s

    Notes
    -----
    Points of deviation from the original implementation and their reasons:

    -

    .. [1] W. Zijlstra, & A. L. Hof, "Assessment of spatio-temporal gait parameters from trunk accelerations during
        human walking" Gait & posture, vol. 18, no. 2, pp. 1-10, 2003.
    .. [2] A. Soltani, et al. "Algorithms for walking speed estimation using a lower-back-worn inertial sensor:
        A cross-validation on speed ranges." IEEE TNSRE, vol 29, pp. 1955-1964, 2021.
    .. [3] https://github.com/mobilise-d/Mobilise-D-TVS-Recommended-Algorithms/blob/master/SLA_SLB/STRIDELEN.m

    """

    beta: float

    def __init__(self, *, beta: float = 0.1) -> None:
        self.beta = beta

    @base_sl_docfiller
    def calculate(self, data: pd.DataFrame, *, initial_contacts: pd.DataFrame, sensor_height: float, sampling_rate_hz: float, tuning_coefficient: float, orientation_method = MadgwickAHRS(beta=0.1), **_: Unpack[dict[str, Any]]) -> Self: # TODO: which default value for orientation method?
        """%(detect_short)s.

        %(detect_info)s

        Parameters
        ----------
        %(detect_para)s

        %(detect_return)s

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # 1. Sensor alignment (optional): Madgwick complementary filter
        newAcc = data[['acc_x', 'acc_y', 'acc_z']] # consider acceleration
        if orientation_method != False: # TODO: how to make rotation optional?
            # perform rotation
            rotated_data = rotate_dataset_series(data, orientation_method.estimate(data,sampling_rate_hz=sampling_rate_hz).orientation_object_[:-1])
            newAcc = rotated_data[['acc_x','acc_y','acc_z']] # consider acceleration

        # 2. High-pass filtering --> lower cut-off: 0.1 Hz, filter design: Butterworth IIR, filter order: 4
        vacc = newAcc['acc_x'] # TODO: check that acceleration is passed to the calculate method in m/s^2
        fc = 0.1 # cut-off frequency
        [df, cf] = butter(4, fc / (sampling_rate_hz / 2), 'high') # design high-pass filter
        vacc_high = lfilter(df, cf, vacc) # filtering

        # 3. Integration of vertical acceleration --> vertical speed
        # 4. Drift removal (high-pass filtering) --> lower cut-off: 1 Hz, filter design: Butterworth IIR, filter order: 4
        # 5. Integration of vertical speed --> vertical displacement d(t)
        # 6. Compute total vertical displacement during the step (d_step):
        # d_step = |max(d(t)) - min(d(t))|
        # 7. Biomechanical model:
        # StepLength = A * 2 * sqrt(2 * LBh * d_step - d_step^2)
        # A: tuning coefficient, optimized by training for each population
        # LBh: sensor height in meters, representative of the height of the center of mass
        #TODO: Which is the correct way to pass tuning_coefficient, initial_contacts, sensor_height?
        sl_zjilstra_v3 = zjilsV3(vacc_high, sampling_rate_hz, tuning_coefficient, initial_contacts, sensor_height)

        # 8. Remove step length outliers during the gait sequence --> Hampel filter based on median absolute deviation
        # 9. Interpolation of StepLength values --> Length per second values
        # 10. Remove length per second outliers during the gait sequence --> Hampel filter based on median absolute deviation
        duration = int(np.floor((len(vacc) / sampling_rate_hz))) # bottom-rounded duration (s)
        HStime = initial_contacts / sampling_rate_hz # initial contacts: samples -> sec
        slSec_zjilstra_v3 = stride2sec(HStime.to_numpy(), duration, sl_zjilstra_v3)
        slmat = slSec_zjilstra_v3[0:duration]
        self.sl_list_ = slmat

        return self


def zjilsV3(LB_vacc_high: np.ndarray, fs: int, K: float, HSsamp: pd.Series, LBh: float) -> np.ndarray:
    # Step length estimation using the biomechanical model propose by Zijlstra & Hof [1]
    # [1] Zijlstra, W., & Hof, A. L. (2003). Assessment of spatio-temporal gait parameters from trunk accelerations
    # during human walking. Gait & posture, 18(2), 1-10.
    # Inputs:
    # - LB_vacc_high: vertical acceleration recorded on lower back, high-pass filtered.
    # - fs: sampling frequency of the acc signal.
    # - K: correction factor estimated by data from various clinical populations
    # - HSsamp: vector containing the timing of heel strikes (or initial contacts) events (in samples)
    # - LBh: Low Back height, i.e., the distance from ground to sensor location on lower back (in m)
    #
    # Output:
    # - sl_zjilstra_v3: estimated step length.

    # estimate vertical speed
    vspeed = -np.cumsum(LB_vacc_high) / fs
    # drift removal (high pass filtering)
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
        # the k-th stride length value is initially estimated as the absolute difference between the maximum and the
        # minimum (range) of the vertical displacement between the k-th and (k+1)-th IC
        h_jilstra_v3[k] = np.abs(max(vdis_high_v2[HSsamp[k]:HSsamp[k + 1]]) -
                                  min(vdis_high_v2[HSsamp[k]:HSsamp[k + 1]]))
    # biomechanical model formula
    sl_zjilstra_v3 = K * np.sqrt(np.abs((2 * LBh * h_jilstra_v3) - (h_jilstra_v3 ** 2)))
    return sl_zjilstra_v3

def stride2sec(ICtime: np.ndarray, duration: int, stl: np.ndarray) -> np.ndarray:
# Function to interpolate step length values to length per second.
# Inputs:
# - ICtime: vector containing the timing of heel strikes (or initial contacts) events (in seconds)
# - duration: bottom-rounded duration of the WB
# - stl: step length values (in m)
#
# Output:
# - stSec: length values per second.


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