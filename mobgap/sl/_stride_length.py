import matplotlib.pyplot as plt
import numpy as np
from hampel import hampel
import pandas as pd
from sklearn.decomposition import PCA
from scipy.signal import butter, filtfilt, lfilter, medfilt
from gaitmap.trajectory_reconstruction.orientation_methods import MadgwickAHRS
from gaitmap.utils.rotations import rotate_dataset_series
from mobgap.data import LabExampleDataset
from mobgap.sl.base import BaseSlDetector, base_sl_docfiller

@base_sl_docfiller
class SlZijlstra(BaseSlDetector): # NOTE: Shouldn't it be called "Estimator" rather than "Detector"?
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
    8. 


    This is based on the implementation published as part of the mobilised project [3]_.
    However, this implementation deviates from the original implementation in some places.
    For details, see the notes section and the examples.

    Parameters
    ----------
    pre_filter
        A pre-processing filter to apply to the data before the icd algorithm is applied.
    cwt_width
        The width of the wavelet

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(ic_list_)s

    Notes
    -----
    Points of deviation from the original implementation and their reasons:

    - We use a different downsampling method, which should be "more" correct from a signal theory perspective,
      but will yield slightly different results.
    - We use a slightly different approach when it comes to the detection of the peaks between the zero crossings.
      However, the results of this step are identical to the matlab implementation.

    .. [1] W. Zijlstra, & A. L. Hof, "Assessment of spatio-temporal gait parameters from trunk accelerations during
        human walking" Gait & posture, vol. 18, no. 2, pp. 1-10, 2003.
    .. [2] A. Soltani, et al. "Algorithms for walking speed estimation using a lower-back-worn inertial sensor:
        A cross-validation on speed ranges." IEEE TNSRE, vol 29, pp. 1955-1964, 2021.
    .. [3] https://github.com/mobilise-d/Mobilise-D-TVS-Recommended-Algorithms/blob/master/SLA_SLB/STRIDELEN.m

    """

    pre_filter: BaseFilter
    cwt_width: float

    _INTERNAL_FILTER_SAMPLING_RATE_HZ: int = 40

    def __init__(self, *, pre_filter: BaseFilter = cf(EpflDedriftedGaitFilter()), cwt_width: float = 9.0) -> None:
        self.pre_filter = pre_filter
        self.cwt_width = cwt_width

    @base_icd_docfiller
    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_: Unpack[dict[str, Any]]) -> Self:
        """%(detect_short)s.

        %(detect_info)s

        Parameters
        ----------
        %(detect_para)s

        %(detect_return)s

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # 0. SELECT RELEVANT COLUMNS
        # For the icd algorithm only vertical acceleration (i.e. x-component) is required.
        relevant_columns = ["acc_x"]  # acc_x: vertical acceleration
        acc_v = data[relevant_columns]
        acc_v = acc_v.to_numpy()

        # 1. RESAMPLING
        signal_downsampled = (
            Resample(self._INTERNAL_FILTER_SAMPLING_RATE_HZ)
            .transform(data=acc_v, sampling_rate_hz=sampling_rate_hz)
            .transformed_data_.squeeze()
        )
        # 2. BAND-PASS FILTERING
        # Padding for short data: this is required because the length of the input vector must be greater
        # than padlen argument of the filtfilt function (3*max(b,a) = 306 by deafult), otherwise a
        # ValueError is raised.
        n_coefficients = len(EpflGaitFilter().coefficients[0])
        len_pad = 4 * n_coefficients
        signal_downsampled_padded = np.pad(signal_downsampled, (len_pad, len_pad), "wrap")
        # The clone() method is used in order to ensure that the new second instance is independent
        acc_v_40_bpf = (
            self.pre_filter.clone()
            .filter(signal_downsampled_padded, sampling_rate_hz=self._INTERNAL_FILTER_SAMPLING_RATE_HZ)
            .filtered_data_.squeeze()
        )
        # Remove the padding
        acc_v_40_bpf_rmzp = acc_v_40_bpf[len_pad - 1 : -len_pad]
        # 3. CUMULATIVE INTEGRAL
        acc_v_lp_int = cumtrapz(acc_v_40_bpf_rmzp, initial=0) / self._INTERNAL_FILTER_SAMPLING_RATE_HZ
        # 4. CONTINUOUS WAVELET TRANSFORM (CWT)
        acc_v_lp_int_cwt, _ = cwt(
            acc_v_lp_int.squeeze(),
            [self.cwt_width],
            "gaus2",
            sampling_period=1 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ,
        )
        acc_v_lp_int_cwt = acc_v_lp_int_cwt.squeeze()
        # Remove the mean from accVLPIntCwt
        acc_v_lp_int_cwt -= acc_v_lp_int_cwt.mean()

        # 5. INTRA-ZERO-CROSSINGS PEAK DETECTION
        # Detect the extrema between the zero crossings
        icd_array = _find_minima_between_zero_crossings(acc_v_lp_int_cwt)

        detected_ics = pd.DataFrame({"ic": icd_array}).rename_axis(index="step_id")
        detected_ics_unsampled = (
            (detected_ics * sampling_rate_hz / self._INTERNAL_FILTER_SAMPLING_RATE_HZ).round().astype(int)
        )
        self.ic_list_ = detected_ics_unsampled

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