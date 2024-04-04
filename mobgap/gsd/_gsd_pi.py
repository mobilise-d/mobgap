import warnings
from itertools import groupby
from typing import Any

import numpy as np
import pandas as pd
import scipy
from gaitmap.utils.array_handling import merge_intervals
from intervaltree import IntervalTree
from typing_extensions import Self, Unpack

from mobgap.data_transform import (
    CwtFilter,
    EpflDedriftedGaitFilter,
    EpflGaitFilter,
    GaussianFilter,
    Pad,
    Resample,
    SavgolFilter,
    chain_transformers,
)
from mobgap.gsd.base import BaseGsDetector, base_gsd_docfiller

# TODO: Add tests (Metatests, some basic tests to trigger most of the error cases, some regression tests on the real
#       world data, some simple tests on artificial data (e.g. no GS detected if just 0 input)
# TODO: Potentially rework find_pulse_trains
# TODO: Complete narrative example including comparison with original Matlab and Ground Truth. See GSD-A example
# TODO: Add version without hilbert transform


@base_gsd_docfiller
class GsdParaschivIonescu(BaseGsDetector):
    """Implementation of the GSD algorithm by Paraschiv-Ionescu et al. (2014) [1]_.

    This algorithm is for the detection of gait (walking) sequences using body acceleration recorded with a tri-axial
    accelerometer worn/fixed on the lower back (close to body center of mass). The algorithm was developed and validated
    using data recorded in patients with impaired mobility (Parkinson's disease, multiple sclerosis, hip fracture,
    post-stroke and cerebral palsy).

    The algorithm detects the gait sequences based on identified steps. In order to enhance the step-related features
    (peaks in acceleration signal), the "active" periods potentially corresponding to locomotion are roughly detected
    and the statistical distribution of the amplitude of the peaks in these active periods is used to derive an adaptive
    (data-driven) threshold for detection of step-related peaks.
    Consecutive steps are associated into gait sequences [1]_ [2]_.

    This is based on the implementation published as part of the mobilised project [3]_.
    However, this implementation deviates from the original implementation in some places.
    For details, see the notes section.

    Note that this algorithm is referred as GSDB in the validation study [4]_ and in the original implementation.

    Abbreviations:
    gs = gait sequence

    Parameters
    ----------
    min_n_steps
        The minimum number of steps allowed in a gait sequence (walking bout).
        Only walking bouts with equal or more detected steps are considered for further processing.
    active_signal_fallback_threshold
        An upper threshold applied to the filtered signal. Minima and maxima beyond this threshold are considered as
        detected steps.
    padding
        A float multiplied by the mean of the step times to pad the start and end of the detected gait sequences.
        The gait sequences are filtered again by number of steps after this padding, removing any gs with too few steps.
    max_gap_s
        Maximum time (in seconds) between consecutive gait sequences.
        If a gap is smaller than max_gap_s, the two consecutive gait sequences are merged into one.
        This is applied after the gait sequences are detected.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(gs_list_)s
        A dataframe containing the start and end times of the detected gait sequences.
        Each row corresponds to a single gs.

    Notes
    -----
    Points of deviation from the original implementation and their reasons:

    - All parameters and thresholds are converted the units used in mobgap.
      Specifically, we use m/s^2 instead of g.
    - We introduced a try/except incase no active periods were detected.
    - In original implementation, stages for filtering by minimum number of steps are hardcoded as:

      - min_n_steps>=4 after find_pulse_trains(MaxPeaks) and find_pulse_trains(MinPeaks)
      - min_n_steps>=3 during the gs padding (NOTE: not implemented in this algorithm since it is redundant here)
      - min_n_steps>=5 before merging gait sequences if time (in seconds) between consecutive gs is smaller than max_gap_s

      This means that original implementation cannot be perfectly replicated with definition of min_n_steps

    - The original implementation used a check for overlapping gait sequences.
      We removed this step since it should not occur.


    .. [1] Paraschiv-Ionescu, A, Soltani A, and Aminian K. "Real-world speed estimation using single trunk IMU:
       methodological challenges for impaired gait patterns." 2020 42nd Annual International Conference of the IEEE
       Engineering in Medicine & Biology Society (EMBC). IEEE, 2020.
    .. [2] Paraschiv-Ionescu, A, et al. "Locomotion and cadence detection using a single trunk-fixed accelerometer:
       validity for children with cerebral palsy in daily life-like conditions." Journal of neuroengineering and
       rehabilitation 16.1 (2019): 1-11.
    .. [3] https://github.com/mobilise-d/Mobilise-D-TVS-Recommended-Algorithms/blob/master/GSDA/Library/GSD_Iluz.m
    .. [4] Micó-Amigo, M. E., Bonci, T., Paraschiv-Ionescu, A., Ullrich, M., Kirk, C., Soltani, A., ... & Del Din,
       S. (2022). Assessing real-world gait with digital technology? Validation, insights and recommendations from the
       Mobilise-D consortium.

    """

    min_n_steps: int
    active_signal_fallback_threshold: float
    max_gap_s: float
    # TODO: Padding is in multiples of the average step time of the respective gait sequence.
    padding: float

    _INTERNAL_FILTER_SAMPLING_RATE_HZ: int = 40

    def __init__(
        self,
        *,
        min_n_steps: int = 5,
        active_signal_fallback_threshold: float = 0.15,
        max_gap_s: float = 3,
        padding: float = 0.75,
    ) -> None:
        self.min_n_steps = min_n_steps
        self.active_signal_fallback_threshold = active_signal_fallback_threshold
        self.max_gap_s = max_gap_s
        self.padding = padding

    @base_gsd_docfiller
    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_: Unpack[dict[str, Any]]) -> Self:
        """%(detect_short)s.

        Parameters
        ----------
        %(detect_para)s

        %(detect_return)s

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        acc = data[["acc_x", "acc_y", "acc_z"]].to_numpy()

        # Signal vector magnitude
        acc_norm = np.linalg.norm(acc, axis=1)

        '''# TODO: Check new filter implementation below is correct

        # We need to initialize the filter once to get the number of coefficients to calculate the padding.
        # This is not ideal, but works for now.
        # TODO: We should evaluate, if we need the padding at all, or if the filter methods that we use handle that
        #  correctly anyway. -> filtfilt uses padding automatically and savgol allows to actiavte padding, put uses the
        #  default mode (polyinomal interpolation) might be suffiecent anyway, cwt might have some edeeffects, but
        #  usually nothing to worry about.
        n_coefficients = len(EpflGaitFilter().coefficients[0])

        #   CWT - Filter
        #   Original MATLAB code calls old version of cwt (open wavelet.internal.cwt in MATLAB to inspect) in
        #   accN_filt3=cwt(accN_filt2,10,'gaus2',1/40);
        #   Here, 10 is the scale, gaus2 is the second derivative of a Gaussian wavelet, aka a Mexican Hat or Ricker
        #   wavelet.
        #   This frequency this scale corresponds to depends on the sampling rate of the data.
        #   As the mobgap cwt method uses the center frequency instead of the scale, we need to calculate the
        #   frequency that scale corresponds to at 40 Hz sampling rate.
        #   Turns out that this is 1.2 Hz
        cwt = CwtFilter(wavelet="gaus2", center_frequency_hz=1.2)

        # Savgol filters
        # The original Matlab code useses two savgol filter in the chain.
        # To replicate them with our classes we need to convert the sample-parameters of the original matlab code to
        # sampling-rate independent units used for the parameters of our classes.
        # The parameters from the matlab code are: (21, 7) and (11, 5)
        savgol_1_win_size_samples = 21
        savgol_1 = SavgolFilter(
            window_length_s=savgol_1_win_size_samples / self._INTERNAL_FILTER_SAMPLING_RATE_HZ,
            polyorder_rel=7 / savgol_1_win_size_samples,
        )
        savgol_2_win_size_samples = 11
        savgol_2 = SavgolFilter(
            window_length_s=savgol_2_win_size_samples / self._INTERNAL_FILTER_SAMPLING_RATE_HZ,
            polyorder_rel=5 / savgol_2_win_size_samples,
        )

        # Now we build everything together into one filter chain.
        filter_chain = [
            # Resample to 40Hz to process with filters
            ("resampling", Resample(self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
            ("savgol_1", savgol_1),
            ("epfl_gait_filter", EpflDedriftedGaitFilter()),
            ("cwt_1", cwt),
            ("savol_2", savgol_2),
            ("cwt_2", cwt),
            ("gaussian_1", GaussianFilter(sigma_s=2 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
            ("gaussian_2", GaussianFilter(sigma_s=2 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
            ("gaussian_3", GaussianFilter(sigma_s=3 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
            ("gaussian_4", GaussianFilter(sigma_s=2 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
        ]

        acc_filtered = chain_transformers(acc_norm, filter_chain, sampling_rate_hz=sampling_rate_hz)
        '''

        # Resample to algorithm_target_fs
        acc_norm_resampled = (
            Resample(self._INTERNAL_FILTER_SAMPLING_RATE_HZ)
            .transform(acc_norm, sampling_rate_hz=sampling_rate_hz)
            .transformed_data_
        )

        # Filter to enhance the acceleration signal, when low SNR, impaired, asymmetric and slow gait
        acc_filtered = scipy.signal.savgol_filter(acc_norm_resampled, polyorder=7,
                                                  window_length=21)  # window_length and polyorder from MATLAB
        acc_filtered = EpflDedriftedGaitFilter().filter(acc_filtered, sampling_rate_hz=40).filtered_data_
        acc_filtered = scipy.signal.cwt(acc_filtered.squeeze(), scipy.signal.ricker,
                                        [7])  # scale=[7] in python matches 10 in MATLAB
        acc_filtered4 = scipy.signal.savgol_filter(acc_filtered, 11, 5)  # window_length and polyorder from MATLAB
        acc_filtered = scipy.signal.cwt(acc_filtered4.squeeze(), scipy.signal.ricker,
                                        [7])  # scale=[7] in python matches 10 in MATLAB
        acc_filtered = scipy.ndimage.gaussian_filter(acc_filtered.squeeze(), 2)  # windowWidth=10 in MATLAB
        acc_filtered = scipy.ndimage.gaussian_filter(acc_filtered.squeeze(), 2)  # windowWidth=10 in MATLAB
        acc_filtered = scipy.ndimage.gaussian_filter(acc_filtered.squeeze(), 3)  # windowWidth=15 in MATLAB
        acc_filtered = scipy.ndimage.gaussian_filter(acc_filtered.squeeze(), 2)  # windowWidth=10 in MATLAB













        try:
            active_peak_threshold = find_active_period_peak_threshold(
                acc_filtered, self._INTERNAL_FILTER_SAMPLING_RATE_HZ
            )
            signal = acc_filtered
        except NoActivePeriodsDetectedError:
            # If we don't find the active periods, use a fallback threshold and use a less filtered signal for further
            # processing, for which we can better predict the threshold.
            warnings.warn("No active periods detected, using fallback threshold", stacklevel=1)
            active_peak_threshold = self.active_signal_fallback_threshold
            x = 0/0
            '''fallback_filter_chain = [
                ("resampling", Resample(self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
                ("savgol_1", savgol_1,),
                ("epfl_gait_filter", EpflDedriftedGaitFilter()),
                ("cwt_1", cwt),
                ("savol_2", savgol_2,),
            ]
            signal = chain_transformers(acc_norm, fallback_filter_chain, sampling_rate_hz=sampling_rate_hz)
            '''

        # Find extrema in signal that might represent steps
        min_peaks, max_peaks = find_min_max_above_threshold(signal, active_peak_threshold)

        # Combine steps detected by the maxima and minima
        gs_from_max = find_pulse_trains(max_peaks)
        gs_from_min = find_pulse_trains(min_peaks)
        gs_from_max = gs_from_max[gs_from_max[:, 2] >= self.min_n_steps]
        gs_from_min = gs_from_min[gs_from_min[:, 2] >= self.min_n_steps]

        # Combine the gs from the maxima and minima
        combined_final = find_intersections(gs_from_max[:, :2], gs_from_min[:, :2])

        # Check if all gs removed
        if combined_final.size == 0:
            self.gs_list_ = pd.DataFrame(columns=["start", "end"]).astype(int)  # Return empty df if no gs
            return self

        # Find all max_peaks within each final gs
        steps_per_gs = [[x for x in max_peaks if gs[0] <= x <= gs[1]] for gs in combined_final]
        n_steps_per_gs = np.array([len(steps) for steps in steps_per_gs])
        mean_step_times = np.array([np.mean(np.diff(steps)) for steps in steps_per_gs])

        # Pad each gs by padding*mean_step_times before and after
        combined_final[:, 0] = np.fix(combined_final[:, 0] - self.padding * mean_step_times)
        combined_final[:, 1] = np.fix(combined_final[:, 1] + self.padding * mean_step_times)

        # Filter again by number of steps, remove any gs with too few steps
        combined_final = combined_final[n_steps_per_gs >= self.min_n_steps]

        if combined_final.size == 0:  # Check if all gs removed
            self.gs_list_ = pd.DataFrame(columns=["start", "end"]).astype(int)  # Return empty df if no gs
            return self

        # Merge gs if time (in seconds) between consecutive gs is smaller than max_gap_s
        combined_final = merge_intervals(
            combined_final, int(np.round(self._INTERNAL_FILTER_SAMPLING_RATE_HZ * self.max_gap_s))
        )

        # Convert back to original sampling rate
        combined_final = combined_final * sampling_rate_hz / self._INTERNAL_FILTER_SAMPLING_RATE_HZ

        # Cap the start and the end of the signal using clip, incase padding extended any gs past the signal length
        combined_final = np.clip(combined_final, 0, len(acc))

        # Compile the df
        self.gs_list_ = pd.DataFrame(combined_final, columns=["start", "end"]).astype(int)

        return self


def hilbert_envelop(sig: np.ndarray, smooth_window: int, duration: int) -> np.ndarray:
    """Apply hilbert transform to select dynamic threshold for activity detection.

    Calculates the analytical signal with the help of hilbert transform, takes the envelope and smooths the signal.
    Finally, with the help of an adaptive threshold detects the activity of the signal where at least a minimum number
    of samples with the length of duration samples should stay above the threshold. The threshold is a computation of
    signal noise and activity level which is updated online.

    Parameters
    ----------
    sig
        A 1D numpy array representing the signal.
    smooth_window
        This is the window length used for smoothing the input signal in terms of number of samples.
    duration
        Number of samples in the window used for updating the threshold.

    Returns
    -------
    active
        A binary 1D numpy array, same length as sig, where 1 represents active periods and 0 represents non-active
        periods.

    .. [1] Sedghamiz, H. BioSigKit: A Matlab Toolbox and Interface for Analysis of BioSignals Software • Review •
        Repository Archive. J. Open Source Softw. 2018, 3, 671

    """
    # Calculate the analytical signal and get the envelope
    amplitude_envelope = np.abs(scipy.signal.hilbert(sig))

    # Take the moving average of analytical signal
    env = np.convolve(
        amplitude_envelope,
        np.ones(smooth_window) / smooth_window,
        "same",  # Smooth
    )
    env -= np.mean(env)  # Get rid of offset
    env /= np.max(env)  # Normalize

    threshold_sig = 4 * np.nanmean(env)
    noise = np.mean(env) / 3  # Noise level
    threshold = np.mean(env)  # Signal level
    update_threshold = False

    # Initialize Buffers
    noise_buff = np.zeros(len(env) - duration + 1)
    active = np.zeros(len(env))

    if np.isnan(threshold_sig):
        return active

    # TODO: This adaptive threshold might be possible to be replaced by a call to find_peaks.
    #       We should test that out once we have a proper evaluation pipeline.
    for i in range(len(env) - duration + 1):
        # Update threshold 10% of the maximum peaks found
        window = env[i : i + duration]
        if (window > threshold_sig).all():
            active[i] = max(env)
            threshold = 0.1 * np.mean(window)
            update_threshold = True
        elif np.mean(window) < threshold_sig:
            noise = np.mean(window)
        elif noise_buff.any():
            noise = np.mean(noise_buff)
        else:
            raise ValueError("Error in threshold update")

        noise_buff[i] = noise

        # Update threshold
        if update_threshold:
            threshold_sig = noise + 0.50 * (abs(threshold - noise))

    return active


def find_min_max_above_threshold(signal: np.ndarray, threshold: float) -> tuple:
    """
    Identify the indices of local minima and maxima  where the values are beyond a specified threshold.

    Parameters
    ----------
    signal
        A 1D numpy array representing the signal.
    threshold
        A threshold value to filter the minima and maxima.

    Returns
    -------
    tuple: Two numpy arrays containing the indices of local minima and maxima, respectively.
    """
    signal = signal.squeeze()
    diff = np.diff(signal)
    extrema_indices = np.nonzero(diff[1:] * diff[:-1] <= 0)[0] + 1

    minima = extrema_indices[diff[extrema_indices] >= 0]
    maxima = extrema_indices[diff[extrema_indices] < 0]

    minima = minima[signal[minima] < -threshold]
    maxima = maxima[signal[maxima] > threshold]

    return minima, maxima


def find_pulse_trains(x: np.ndarray) -> np.ndarray:
    walkflag = 0
    thd = 3.5 * 40
    n = 0

    start = [0]
    steps = [0]
    end = [0]

    if len(x) > 2:
        for i in range(len(x) - 1):
            if x[i + 1] - x[i] < thd:
                if walkflag == 0:
                    start[n] = x[i]
                    steps[n] = 1
                    walkflag = 1
                else:
                    steps[n] = steps[n] + 1
                    thd = 1.5 * 40 + (x[i] - start[n]) / steps[n]
            elif walkflag == 1:
                end[n] = x[i - 1]
                n = n + 1
                start = [*start, 0]
                steps = [*steps, 0]
                end = [*end, 0]
                walkflag = 0
                thd = 3.5 * 40

    if walkflag == 1:
        if x[-1] - x[-2] < thd:
            end[-1] = x[-1]
            steps[n] = steps[n] + 1
        else:
            end[-1] = x[-1]

    return np.array([start, end, steps]).T


def find_intersections(intervals_a: np.ndarray, intervals_b: np.ndarray) -> np.ndarray:
    """Find the intersections between two sets of intervals.

    Parameters
    ----------
    intervals_a
        The first list of intervals. Each interval is represented as a tuple of two integers.
    intervals_b
        The second list of intervals. Each interval is represented as a tuple of two integers.

    Returns
    -------
    np.ndarray
        An array of intervals that are the intersections of the intervals in `intervals_a` and `intervals_b`.
        Each interval is represented as a list of two integers.

    """
    # Create Interval Trees
    intervals_a_tree = IntervalTree.from_tuples(intervals_a)
    intervals_b_tree = IntervalTree.from_tuples(intervals_b)

    overlap_intervals = []

    # Calculate TP and FP
    for interval in intervals_b_tree:
        overlaps = sorted(intervals_a_tree.overlap(interval.begin, interval.end))
        if overlaps:
            for overlap in overlaps:
                start = max(interval.begin, overlap.begin)
                end = min(interval.end, overlap.end)
                overlap_intervals.append([start, end])

    return merge_intervals(np.array(overlap_intervals)) if len(overlap_intervals) != 0 else np.array([])


class NoActivePeriodsDetectedError(Exception):
    pass


def find_active_period_peak_threshold(signal: np.ndarray, sampling_rate_hz: int) -> float:
    # Find pre-detection of 'active' periods in order to estimate the amplitude of acceleration peaks
    alarm = hilbert_envelop(signal, sampling_rate_hz, sampling_rate_hz)

    if not np.any(alarm):
        raise NoActivePeriodsDetectedError()

    # TODO: What does all of this do?
    # Length of each consecutive stretch of nonzero values in alarm
    len_alarm = [len(list(s)) for v, s in groupby(alarm, key=lambda x: x > 0)]
    end_alarm = np.cumsum(len_alarm)
    start_alarm = np.concatenate([np.array([0]), end_alarm[:-1]])
    # Whether each consecutive stretch of nonzero values in alarm is alarmed
    alarmed = [v for v, s in groupby(alarm, key=lambda x: x > 0)]

    walk = np.array([])  # Initialise detected periods of walking variable
    for s, e, a in zip(start_alarm, end_alarm, alarmed):  # Iterate through the consecutive periods
        if a:  # If alarmed
            if e - s <= 3 * sampling_rate_hz:  # If the length of the alarm period is too short
                alarm[s:e] = 0  # Replace this section of alarm with zeros
            else:
                walk = np.concatenate([walk, signal[s - 1 : e - 1]])

    if walk.size == 0:
        raise NoActivePeriodsDetectedError()

    peaks_p, _ = scipy.signal.find_peaks(walk)
    peaks_n, _ = scipy.signal.find_peaks(-walk)
    pksp, pksn = walk[peaks_p], -walk[peaks_n]
    pks = np.concatenate([pksp[pksp > 0], pksn[pksn > 0]])
    return np.percentile(pks, 5)  # Data adaptive threshold
