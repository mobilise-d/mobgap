import warnings
from typing import Any

import numpy as np
import pandas as pd
from intervaltree import IntervalTree
from numba import njit
from scipy.signal import find_peaks, hilbert
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc
from mobgap._utils_internal.misc import timed_action_method
from mobgap.consts import BF_ACC_COLS, GRAV_MS2, SF_ACC_COLS
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
from mobgap.gait_sequences.base import BaseGsDetector, _unify_gs_df, base_gsd_docfiller
from mobgap.utils.array_handling import (
    bool_array_to_start_end_array,
    merge_intervals,
    start_end_array_to_bool_array,
)
from mobgap.utils.conversions import as_samples
from mobgap.utils.dtypes import get_frame_definition

_gsd_ionescu_docfiller = make_filldoc(
    base_gsd_docfiller._dict
    | {
        "common_parameters": """
    min_n_steps
        The minimum number of steps allowed in a gait sequence (walking bout).
        Only walking bouts with equal or more detected steps are considered for further processing.
    padding
        A float multiplied by the mean of the step times to pad the start and end of the detected gait sequences.
        The gait sequences are filtered again by number of steps after this padding, removing any gs with too few steps.
    max_gap_s
        Maximum time (in seconds) between consecutive gait sequences.
        If a gap is smaller than max_gap_s, the two consecutive gait sequences are merged into one.
        This is applied after the gait sequences are detected.
    min_step_margin_s
        The minimum time margin (in seconds) between two consecutive initial contacts within a gait sequence.
        This is used when combining consecutive steps candidates into gait sequences.
        The actual threshold is calculated as the mean of the step times plus this parameter.
""",
        "coordinate_system_note": """
    .. note:: Compared to other GSD algorithms, this algorithm only works on the Acc norm and hence, can be used with
    out knowing the previous alignment of the sensor.
    The algorithm therefore allows to pass data in either the body or the sensor frame.
""",
    }
)


class _BaseGsdIonescu(BaseGsDetector):
    min_n_steps: int
    active_signal_threshold: float
    max_gap_s: float
    min_step_margin_s: float
    padding: float

    _INTERNAL_FILTER_SAMPLING_RATE_HZ: int = 40

    def __init__(
        self,
        *,
        min_n_steps: int,
        max_gap_s: float,
        min_step_margin_s: float,
        padding: float,
    ) -> None:
        self.min_n_steps = min_n_steps
        self.max_gap_s = max_gap_s
        self.min_step_margin_s = min_step_margin_s
        self.padding = padding

    def _find_step_candidates(self, acc_norm: np.ndarray, sampling_rate_hz: float) -> tuple:
        raise NotImplementedError()

    @timed_action_method
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

        # As we work on the norm, we support both coordinate system definitions.
        frame = get_frame_definition(data, ["sensor", "body"])

        acc_axis = SF_ACC_COLS if frame == "sensor" else BF_ACC_COLS

        acc = data[acc_axis].to_numpy()

        if not self.min_n_steps >= 1:
            raise ValueError("min_n_steps must be at least 1")

        # Signal vector magnitude
        acc_norm = np.linalg.norm(acc, axis=1)

        # Peaks are in samples based on internal sampling rate
        min_peaks, max_peaks = self._find_step_candidates(acc_norm, sampling_rate_hz)

        # Combine steps detected by the maxima and minima
        allowed_distance_between_peaks = as_samples(self.min_step_margin_s, self._INTERNAL_FILTER_SAMPLING_RATE_HZ)
        step_margin = as_samples(self.min_step_margin_s, self._INTERNAL_FILTER_SAMPLING_RATE_HZ)

        gs_from_max = find_pulse_trains(max_peaks, allowed_distance_between_peaks, step_margin)
        gs_from_min = find_pulse_trains(min_peaks, allowed_distance_between_peaks, step_margin)

        # Combine the gs from the maxima and minima
        combined_final = find_intersections(gs_from_max, gs_from_min)

        # Check if all gs removed
        if combined_final.size == 0:
            self.gs_list_ = _unify_gs_df(pd.DataFrame(columns=["start", "end"]))
            return self

        # Find all max_peaks within each final gs
        steps_per_gs = [[x for x in max_peaks if gs[0] <= x <= gs[1]] for gs in combined_final]
        n_steps_per_gs = np.array([len(steps) for steps in steps_per_gs])
        # It can happen that we only have one step in a gs, in this case we can not calculate the mean step time
        # Numpy will output NaN and throw a warning.
        # We don't want ot see the warning, so we suppress it.
        # GSs that don't have enough steps will be removed later anyway.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_step_times = np.array([np.mean(np.diff(steps)) for steps in steps_per_gs])
            # Pad each gs by padding*mean_step_times before and after
            combined_final[:, 0] = np.fix(combined_final[:, 0] - self.padding * mean_step_times)
            combined_final[:, 1] = np.fix(combined_final[:, 1] + self.padding * mean_step_times)

        # Filter again by number of steps, remove any gs with too few steps
        combined_final = combined_final[n_steps_per_gs >= self.min_n_steps]

        if combined_final.size == 0:  # Check if all gs removed
            self.gs_list_ = _unify_gs_df(pd.DataFrame(columns=["start", "end"]))
            return self

        # Merge gs if time (in seconds) between consecutive gs is smaller than max_gap_s
        combined_final = merge_intervals(
            combined_final, as_samples(self.max_gap_s, self._INTERNAL_FILTER_SAMPLING_RATE_HZ)
        )

        # Convert back to original sampling rate
        combined_final = combined_final * sampling_rate_hz / self._INTERNAL_FILTER_SAMPLING_RATE_HZ

        # Cap the start and the end of the signal using clip, in case padding extended any gs past the signal length
        combined_final = np.clip(combined_final, 0, len(acc))

        # Compile the df
        self.gs_list_ = _unify_gs_df(pd.DataFrame(combined_final, columns=["start", "end"]))

        return self


@_gsd_ionescu_docfiller
class GsdIonescu(_BaseGsdIonescu):
    """Implementation of the GSD algorithm developed by Paraschiv-Ionescu et al. (2014) [1]_.

    .. note:: A version of this algorithm with adaptive threshold is also available as :class:`GsdAdaptiveIonescu`.

    The method defines gait sequences based on the detected steps.
    Steps are detected by identifying local minima and maxima in the filtered acceleration signal that are above a
    specified threshold.
    The outputs are further filtered by the number of steps and consecutive gait sequence with short breaks are merged.

    %(coordinate_system_note)s

    Parameters
    ----------
    active_signal_threshold
        A threshold applied to the filtered acceleration norm.
        Minima and maxima beyond this threshold are considered as detected steps.
        The unit of this threshold is techically m/s^2, but as the signal is heavily filtered the value range can not
        be easily inferred.
        To properly set this threshold, it is recommended to use the ``filtered_signal_`` debug attribute.

        .. note:: In the original code, this threshold was in the unit `g`. So if you are transferring thresholds from
                  the original code, you need to convert them to m/s^2.
    %(common_parameters)s

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(gs_list_)s
        A dataframe containing the start and end times of the detected gait sequences.
        Each row corresponds to a single gs.
    filtered_signal_
        The filtered acceleration norm used for step detection.
    %(perf_)s

    Notes
    -----
    While the signal filtering is based on the original implementation, this implementation adds the post-processing
    steps that originally were only implemented for the adaptive threshold version of the algorithm [2]_.
    We also remove the original "n-step-filter" in favor of this post-processing step.
    Furthermore, we fixed a bug where the average step time during the "pulse train" identification was calculated
    using n + 1 steps.

    .. [1] Paraschiv-Ionescu, A, et al. "Locomotion and cadence detection using a single trunk-fixed accelerometer:
       validity for children with cerebral palsy in daily life-like conditions." Journal of neuroengineering and
       rehabilitation 16.1 (2019): 1-11.
    .. [2] https://github.com/mobilise-d/Mobilise-D-TVS-Recommended-Algorithms/blob/master/GSDB/GSD_LowBackAcc.m

    """

    active_signal_threshold: float

    filtered_signal_: np.ndarray

    def __init__(
        self,
        *,
        min_n_steps: int = 5,
        active_signal_threshold: float = 0.1 * GRAV_MS2,
        max_gap_s: float = 3.5,
        min_step_margin_s: float = 1.5,
        padding: float = 0.75,
    ) -> None:
        self.active_signal_threshold = active_signal_threshold
        super().__init__(
            min_n_steps=min_n_steps, max_gap_s=max_gap_s, padding=padding, min_step_margin_s=min_step_margin_s
        )

    def _find_step_candidates(self, acc_norm: np.ndarray, sampling_rate_hz: float) -> tuple:
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
        # To replicate them with our classes we need to convert the sample-parameters of the original matlab code to
        # sampling-rate independent units used for the parameters of our classes.
        # The parameters from the matlab code are: (1, 3)
        savgol_win_size_samples = 3
        savgol = SavgolFilter(
            window_length_s=savgol_win_size_samples / self._INTERNAL_FILTER_SAMPLING_RATE_HZ,
            polyorder_rel=1 / savgol_win_size_samples,
        )

        # We need to initialize the filter once to get the number of coefficients to calculate the padding.
        # This is not ideal, but works for now.
        n_coefficients = len(EpflGaitFilter().coefficients[0])

        # Padding to cope with short data
        len_pad_s = 4 * n_coefficients / self._INTERNAL_FILTER_SAMPLING_RATE_HZ
        padding = Pad(pad_len_s=len_pad_s, mode="wrap")

        active_peak_threshold = self.active_signal_threshold
        fallback_filter_chain = [
            ("resampling", Resample(self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
            ("padding", padding),
            (
                "savgol_1",
                savgol.clone(),
            ),
            ("epfl_gait_filter", EpflDedriftedGaitFilter()),
            ("cwt", cwt),
            (
                "savol_2",
                savgol.clone(),
            ),
            ("padding_remove", padding.get_inverse_transformer()),
        ]
        signal = chain_transformers(acc_norm, fallback_filter_chain, sampling_rate_hz=sampling_rate_hz)
        self.filtered_signal_ = signal
        # Find extrema in signal that might represent steps
        return find_peaks(-signal, height=active_peak_threshold)[0], find_peaks(signal, height=active_peak_threshold)[0]


@_gsd_ionescu_docfiller
class GsdAdaptiveIonescu(_BaseGsdIonescu):
    """Implementation of the GSD algorithm by Paraschiv-Ionescu et al. (2019) [1, 2]_ with adaptive threshold.

    The algorithm was developed and validated using data recorded in patients with impaired mobility
    (Parkinson's disease, multiple sclerosis, hip fracture, post-stroke and cerebral palsy).

    The algorithm detects the gait sequences based on identified steps. In order to enhance the step-related features
    (peaks in acceleration signal), the "active" periods potentially corresponding to locomotion are roughly detected
    and the statistical distribution of the amplitude of the peaks in these active periods is used to derive an adaptive
    (data-driven) threshold for detection of step-related peaks.
    Consecutive steps are associated into gait sequences [1]_ [2]_.

    This is based on the implementation published as part of the mobilised project [3]_.
    However, this implementation deviates from the original implementation in some places.
    For details, see the notes section.

    %(coordinate_system_note)s

    Parameters
    ----------
    active_signal_fallback_threshold
        An upper threshold applied to the filtered signal. Minima and maxima beyond this threshold are considered as
        detected steps.
    %(common_parameters)s

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(gs_list_)s
        A dataframe containing the start and end times of the detected gait sequences.
        Each row corresponds to a single gs.
    filtered_signal_
        The filtered acceleration norm used for step detection.
        This signal will be different depending on whether the adaptive threshold estimation was successful or not.
    threshold_
        The threshold used for step detection.
        This is either the adaptive threshold or the fallback threshold if no active periods were detected.
    adaptive_threshold_success_
        A boolean indicating whether the adaptive threshold estimation was successful.
    %(perf_)s

    Notes
    -----
    Points of deviation from the original implementation and their reasons:

    - All parameters and thresholds are converted the units used in mobgap.
      Specifically, we use m/s^2 instead of g.
    - We introduced a try/except incase no active periods were detected.
    - We fixed a bug where the average step time during the "pulse train" identification was calculated using n + 1
      steps.
    - In original implementation, stages for filtering by minimum number of steps are hardcoded as:

      - min_n_steps>=4 after find_pulse_trains(MaxPeaks) and find_pulse_trains(MinPeaks)
      - min_n_steps>=3 during the gs padding (NOTE: not implemented in this algorithm since it is redundant here)
      - min_n_steps>=5 before merging gait sequences if time (in seconds) between consecutive gs is smaller
        than max_gap_s

      This means that original implementation cannot be perfectly replicated with definition of min_n_steps

    - The original implementation used a check for overlapping gait sequences.
      We removed this step since it should not occur.


    .. [1] Paraschiv-Ionescu, A, Soltani A, and Aminian K. "Real-world speed estimation using single trunk IMU:
       methodological challenges for impaired gait patterns." 2020 42nd Annual International Conference of the IEEE
       Engineering in Medicine & Biology Society (EMBC). IEEE, 2020.
    .. [2] Paraschiv-Ionescu, A, et al. "Locomotion and cadence detection using a single trunk-fixed accelerometer:
       validity for children with cerebral palsy in daily life-like conditions." Journal of neuroengineering and
       rehabilitation 16.1 (2019): 1-11.
    .. [3] https://github.com/mobilise-d/Mobilise-D-TVS-Recommended-Algorithms/blob/master/GSDB/GSD_LowBackAcc.m

    """

    filtered_signal_: np.ndarray
    threshold_: float
    adaptive_threshold_success_: bool

    active_signal_fallback_threshold: float

    def __init__(
        self,
        *,
        min_n_steps: int = 5,
        active_signal_fallback_threshold: float = 0.15 * GRAV_MS2,
        max_gap_s: float = 3.5,
        min_step_margin_s: float = 1.5,
        padding: float = 0.75,
    ) -> None:
        self.active_signal_fallback_threshold = active_signal_fallback_threshold
        super().__init__(
            min_n_steps=min_n_steps, max_gap_s=max_gap_s, padding=padding, min_step_margin_s=min_step_margin_s
        )

    def _find_step_candidates(self, acc_norm: np.ndarray, sampling_rate_hz: float) -> tuple:
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
        # The original Matlab code uses two savgol filter in the chain.
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

        # We need to initialize the filter once to get the number of coefficients to calculate the padding.
        # This is not ideal, but works for now.
        n_coefficients = len(EpflGaitFilter().coefficients[0])

        # Padding to cope with short data
        len_pad_s = 4 * n_coefficients / self._INTERNAL_FILTER_SAMPLING_RATE_HZ
        padding = Pad(pad_len_s=len_pad_s, mode="wrap")

        # Now we build everything together into one filter chain.
        filter_chain = [
            # Resample to 40Hz to process with filters
            ("resampling", Resample(self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
            ("padding", padding),
            ("savgol_1", savgol_1),
            ("epfl_gait_filter", EpflDedriftedGaitFilter()),
            ("cwt_1", cwt),
            ("savol_2", savgol_2),
            ("cwt_2", cwt),
            ("gaussian_1", GaussianFilter(sigma_s=2 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
            ("gaussian_2", GaussianFilter(sigma_s=2 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
            ("gaussian_3", GaussianFilter(sigma_s=3 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
            ("gaussian_4", GaussianFilter(sigma_s=2 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
            ("padding_remove", padding.get_inverse_transformer()),
        ]

        acc_filtered = chain_transformers(acc_norm, filter_chain, sampling_rate_hz=sampling_rate_hz)

        try:
            active_peak_threshold = find_active_period_peak_threshold(
                acc_filtered,
                hilbert_window_size=as_samples(1, self._INTERNAL_FILTER_SAMPLING_RATE_HZ),
                min_active_period_duration=as_samples(3, self._INTERNAL_FILTER_SAMPLING_RATE_HZ),
            )
            signal = acc_filtered
            self.adaptive_threshold_success_ = True
        except NoActivePeriodsDetectedError:
            self.adaptive_threshold_success_ = False
            # If we don't find the active periods, use a fallback threshold and use a less filtered signal for further
            # processing, for which we can better predict the threshold.
            warnings.warn("No active periods detected, using fallback threshold", stacklevel=1)
            active_peak_threshold = self.active_signal_fallback_threshold
            fallback_filter_chain = [
                ("resampling", Resample(self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
                ("padding", padding),
                (
                    "savgol_1",
                    savgol_1,
                ),
                ("epfl_gait_filter", EpflDedriftedGaitFilter()),
                ("cwt_1", cwt),
                (
                    "savol_2",
                    savgol_2,
                ),
                ("padding_remove", padding.get_inverse_transformer()),
            ]
            signal = chain_transformers(acc_norm, fallback_filter_chain, sampling_rate_hz=sampling_rate_hz)

        self.filtered_signal_ = signal
        self.threshold_ = active_peak_threshold

        # Find extrema in signal that might represent steps
        return find_peaks(-signal, height=active_peak_threshold)[0], find_peaks(signal, height=active_peak_threshold)[0]


def active_regions_from_hilbert_envelop(sig: np.ndarray, smooth_window: int, duration: int) -> np.ndarray:
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
        A binary 1D numpy array, same length as sig, where True represents active periods and False represents
        non-active periods.

    .. [1] Sedghamiz, H. BioSigKit: A Matlab Toolbox and Interface for Analysis of BioSignals Software • Review •
        Repository Archive. J. Open Source Softw. 2018, 3, 671

    """
    # Calculate the analytical signal and get the envelope
    amplitude_envelope = np.abs(hilbert(sig))

    # Take the moving average of analytical signal
    env = np.convolve(
        amplitude_envelope,
        np.ones(smooth_window) / smooth_window,
        "same",  # Smooth
    )

    active = np.zeros(len(env))

    env -= np.mean(env)  # Get rid of offset
    if np.all(env == 0):
        return active.astype(bool)
    env /= np.max(env)  # Normalize

    threshold_sig = 4 * np.nanmean(env)
    noise = np.mean(env) / 3  # Noise level
    threshold = np.mean(env)  # Signal level
    update_threshold = False

    # Initialize Buffers
    noise_buff = np.zeros(len(env) - duration + 1)

    if np.isnan(threshold_sig):
        return active.astype(bool)

    # TODO: This adaptive threshold might be possible to be replaced by a call to find_peaks.
    #       We should test that out once we have a proper evaluation pipeline.
    maxenv = max(env)
    for i in range(len(env) - duration + 1):
        # Update threshold 10% of the maximum peaks found
        window = env[i : i + duration]
        mean_win = np.mean(window)
        if (window > threshold_sig).all():
            active[i] = maxenv
            threshold = 0.1 * mean_win
            update_threshold = True
        elif mean_win < threshold_sig:
            noise = mean_win
        elif noise_buff.any():
            noise = np.mean(noise_buff)
        # NOTE: no else case in the original implementation

        noise_buff[i] = noise

        # Update threshold
        if update_threshold:
            threshold_sig = noise + 0.50 * (abs(threshold - noise))

    return active.astype(bool)


@njit(cache=True)
def _find_pulse_train_end(x: np.ndarray, step_threshold: float) -> np.ndarray:
    start_val = x[0]

    for ic_idx, (current_val, next_val) in enumerate(zip(x[1:], x[2:])):
        # We already know that the first two values belong to the pulse train, as this is determined by the caller so we
        # start everything at index 1
        n_steps = ic_idx + 1
        # We update the threshold to be the mean step time + the step threshold
        # Note: The original implementation uses effectively n_steps + 1 here, which likely a bug, as it counts the
        # number of pulses within the pulse train and not the number of distances between pulses.
        thd_step = (current_val - start_val) / n_steps + step_threshold
        if next_val - current_val > thd_step:
            return x[: n_steps + 1]
    return x


@njit(cache=True)
def find_pulse_trains(
    x: np.ndarray, initial_distance_threshold_samples: float, step_threshold_margin: float
) -> np.ndarray:
    start_ends = []
    i = 0
    while i < len(x) - 1:
        # We search for a start of a pulse train
        # This happens, in case 2 consecutive samples are closer than the initial distance threshold
        if x[i + 1] - x[i] < initial_distance_threshold_samples:
            # Then we search for the end of the pulse train
            # This happens, in case 2 consecutive samples are further apart than the step threshold + the mean step time
            # within the pulse train
            start = x[i]
            pulses = _find_pulse_train_end(x[i:], step_threshold_margin)
            start_ends.append([start, pulses[-1]])
            i += len(pulses)
        else:
            i += 1

    if len(start_ends) == 0:
        return np.empty((0, 2), dtype=np.int32)

    start_ends_array = np.array(start_ends, dtype=np.int32)
    return start_ends_array


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


def find_active_period_peak_threshold(
    signal: np.ndarray, *, min_active_period_duration: int, hilbert_window_size: int
) -> float:
    # Find pre-detection of 'active' periods in order to estimate the amplitude of acceleration peaks
    active_regions = active_regions_from_hilbert_envelop(signal, hilbert_window_size, hilbert_window_size)

    if not np.any(active_regions):
        raise NoActivePeriodsDetectedError()

    active_regions_start_end = bool_array_to_start_end_array(active_regions)
    to_short_active_regions = (
        active_regions_start_end[:, 1] - active_regions_start_end[:, 0]
    ) <= min_active_period_duration
    active_regions_start_end = active_regions_start_end[~to_short_active_regions]

    if len(active_regions_start_end) == 0:
        raise NoActivePeriodsDetectedError()

    final_active_area = signal[start_end_array_to_bool_array(active_regions_start_end, len(active_regions))]

    _, props_p = find_peaks(final_active_area, height=0)
    _, props_n = find_peaks(-final_active_area, height=0)
    pks = np.concatenate([props_p["peak_heights"], props_n["peak_heights"]])
    return np.percentile(pks, 5)  # Data adaptive threshold
