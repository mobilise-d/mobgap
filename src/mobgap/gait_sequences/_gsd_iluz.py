import warnings
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import pandas as pd
from numba import float32, float64, guvectorize, int32
from tpcp import cf
from tpcp.misc import classproperty, set_defaults
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import timed_action_method
from mobgap.consts import GRAV_MS2
from mobgap.data_transform import FirFilter
from mobgap.data_transform.base import BaseFilter
from mobgap.gait_sequences.base import BaseGsDetector, _unify_gs_df, base_gsd_docfiller
from mobgap.utils.array_handling import merge_intervals, sliding_window_view
from mobgap.utils.conversions import as_samples
from mobgap.utils.dtypes import assert_is_sensor_data


@base_gsd_docfiller
class GsdIluz(BaseGsDetector):
    """Implementation of the GSD algorithm by Iluz et al. (2014) [1]_.

    The algorithm identifies steps in overlapping windows by convolving the data with a sin signal.
    Depending on the number of identified peaks, the window is classified as a gait sequence or not.
    Consecutive windows are then merged to form the final gait sequences.

    This is based on the implementation published as part of the mobilised project [2]_.
    However, this implementation deviates from the original implementation in some places.
    For details, see the notes section and the examples.

    Parameters
    ----------
    pre_filter
        A pre-processing filter to apply to the data before the GSD algorithm is applied.
    window_length_s
        The length of the window in seconds that is used to detect gait sequences.
        Each window will be processed separately.
    window_overlap
        The overlap between two consecutive windows in percent.
        For example, a value of 0.5 means that the windows will overlap by 50%%.
    std_activity_threshold
        The lower threshold for the standard deviation of the filtered acc_x data to be considered as activity.
    mean_activity_threshold
        A lower threshold applied to the mean of the mean-shifted raw gravity corrected acc_x data to be considered as
        activity.
    acc_v_standing_threshold
        A lower threshold applied to the mean of the acc_v data in each window to detect standing/upright positions.
        Only "standing" windows are considered for further processing.
    step_detection_thresholds
        The minimal peak height for the step detection.
        This expects a tuple with two values, one for each axis (acc_x and acc_z).
    sin_template_freq_hz
        The frequency of the sin template used for the convolution.
    allowed_steps_per_s
        A tuple with two values, specifying the lower and upper bound for the number of steps per second.
        This is converted in a minimum and maximum number of steps per window using the ``window_length_s`` parameter.
    allowed_acc_v_change_per_window
        The maximum change in the mean of the acc_v data between the first and the last second of the window in percent.
        I.e. 0.1 means a maximum change of 10%%.
        If this change is exceeded, the window is discarded, as we assume that the person changed their posture (i.e
        from lying to standing).
    min_gsd_duration_s
        The minimum duration of a gait sequence in seconds.
        This is applied after the gait sequences are detected.
    use_original_peak_detection
        If True, the original peak detection algorithm is used.
        It uses zero crossings to identify peaks and further interpolate the existence of peaks, when none are found
        for a certain period of time.
        We default to the new peak detection algorithm, as it is simpler and less magic.
        The performance of the two algorithms is similar, but not identical.
        For the best possible performance with the original algorithm, some of the other parameters might need to be
        adjusted.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(gs_list_)s
    %(perf_)s

    Notes
    -----
    Points of deviation from the original implementation and their reasons:

    - The order of processing is changed.
      In the original implementation, steps are detected early on in the pipeline and later further thresholds on the
      raw signal are used to discard certain parts of the signal and non-gait.
      We flip the order to reduce the number of windows we need to apply step detection to.
      This is done, because the step detection process is the most expensive part of the algorithm.
    - Instead of a custom peak detection algorithm, we use the a simple peak detection algorithm based on the version
      implemented in scipy, by default.
      We reimplemented in `numba` for significant speedup.
      This method produces similar but different results.
      Most notably, it does not have a maximal distance parameter and does not "interpolate" peaks, if large gaps
      between peaks are detected.
      This means, that the new method often reduces the number of detected peaks.
      To counter act this, we use a significantly lower signal threshold for the pa-axis.
      The orignal value was 1.2 g, we use 0.5 g.

      A reimplementation of the original implementation can be used by setting `use_original_peak_detection=True`.
      However, we default to the new one, as it is "less" magic.
    - The original implementation uses different lower threshold for the allowed numbers per step.
      I.e. different numbers of peaks are expected for the two axes.
      As this is not mentioned anywhere in the paper, and using the same threshold for both axis did not seem to have a
      negative impact on the results, we decided to use the same threshold for both axes to simplify the algorithm.
    - All parameters and thresholds are converted the units used in mobgap.
      Specifically, we use m/s^2 instead of g.
    - The original implementation used a check, that if the sum of the signal in the window is below the min-height
      threshold no peaks are detected.
      We assume that this is an error and use the max of the signal instead.
    - The filter order of the pre-filter is reduced to 100 from 200, as we use a filtfilt implementation, instead of
      just forward filtering.
    - We normalize the convolution by the sampling rate.
      Otherwise, the amplitude of the convolution would scale with the sampling rate and the thresholds would need to be
      adjusted accordingly.
      This was already a problem in the original version, as the same code was used with 128 Hz and published for the
      use with 100 Hz.

    .. [1] T. Iluz, E. Gazit, T. Herman, E. Sprecher, M. Brozgol, N. Giladi, A. Mirelman, and J. M. Hausdorff,
        “Automated detection of missteps during community ambulation in patients with parkinsons disease: a new
        approach for quantifying fall risk in the community setting,” J Neuroeng Rehabil, vol. 11, no. 1, p. 48, 2014.
    .. [2] https://github.com/mobilise-d/Mobilise-D-TVS-Recommended-Algorithms/blob/master/GSDA/Library/GSD_Iluz.m
    """

    pre_filter: BaseFilter
    window_length_s: float
    window_overlap: float
    std_activity_threshold: float
    mean_activity_threshold: float
    acc_v_standing_threshold: float
    step_detection_thresholds: tuple[float, float]
    sin_template_freq_hz: float
    allowed_steps_per_s: tuple[float, float]
    allowed_acc_v_change_per_window: float
    min_gsd_duration_s: float

    class PredefinedParameters:
        _shared: Final = {
            # Original filter order is 200, but just forward filtering is used.
            # We use a filtfilt (zero phase) implementation. This is roughly equivalent to a filter with double the
            # order.
            "pre_filter": FirFilter(order=100, cutoff_freq_hz=(0.5, 3), filter_type="bandpass"),
            "window_length_s": 3,
            "window_overlap": 0.5,
            "std_activity_threshold": 0.01 * GRAV_MS2,
            "mean_activity_threshold": -0.1 * GRAV_MS2,
            "acc_v_standing_threshold": 0.5 * GRAV_MS2,
            "sin_template_freq_hz": 2,
            # Note: The original implementation uses (window-size - 1)/window-size step per second as the lower bound.
            #       This means a minimum of 2 steps per 3-second window.
            #       We hence use 2/3 steps per second as the lower bound.
            "allowed_steps_per_s": (
                2 / 3,
                3,
            ),
            "allowed_acc_v_change_per_window": 0.15,
            "min_gsd_duration_s": 5,
        }

        @classproperty
        def original(cls) -> dict[str, Any]:  # noqa: N805
            return MappingProxyType(
                cls._shared
                | {
                    "use_original_peak_detection": True,
                    "step_detection_thresholds": (
                        # Note that these thresholds are divided by 128, as the original implementation was designed for
                        # 128 Hz. Our implemenation takes the sampling rate out of the scaling via normalization.
                        0.4 * GRAV_MS2 / 128,
                        1.5 * GRAV_MS2 / 128,
                    ),
                }
            )

        @classproperty
        def updated(cls) -> dict[str, Any]:  # noqa: N805
            return MappingProxyType(
                cls._shared
                | {
                    "use_original_peak_detection": False,
                    "step_detection_thresholds": (
                        # TODO: Find good values for the thresholds
                        0.4 * GRAV_MS2 / 100,
                        # This value is very different to the original implementation.
                        # There a peakheight of 1.5 g was used.
                        # However, for some reason, this did not show good performance in our testing.
                        0.5 * GRAV_MS2 / 100,
                    ),
                }
            )

    @set_defaults(**{k: cf(v) for k, v in PredefinedParameters.updated.items()})
    def __init__(
        self,
        *,
        pre_filter: BaseFilter,
        window_length_s: float,
        window_overlap: float,
        std_activity_threshold: float,
        mean_activity_threshold: float,
        acc_v_standing_threshold: float,
        step_detection_thresholds: tuple[float, float],
        sin_template_freq_hz: float,
        allowed_steps_per_s: tuple[float, float],
        allowed_acc_v_change_per_window: float,
        min_gsd_duration_s: float,
        use_original_peak_detection: bool,
    ) -> None:
        self.window_length_s = window_length_s
        self.window_overlap = window_overlap
        self.std_activity_threshold = std_activity_threshold
        self.mean_activity_threshold = mean_activity_threshold
        self.acc_v_standing_threshold = acc_v_standing_threshold
        self.sin_template_freq_hz = sin_template_freq_hz
        self.pre_filter = pre_filter
        self.allowed_steps_per_s = allowed_steps_per_s
        self.step_detection_thresholds = step_detection_thresholds
        self.allowed_acc_v_change_per_window = allowed_acc_v_change_per_window
        self.min_gsd_duration_s = min_gsd_duration_s
        self.use_original_peak_detection = use_original_peak_detection

    @timed_action_method
    @base_gsd_docfiller
    def detect(  # noqa: PLR0915
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """%(detect_short)s.

        Parameters
        ----------
        %(detect_para)s

        %(detect_return)s

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        assert_is_sensor_data(data, frame="body")

        relevant_columns = ["acc_is", "acc_pa"]
        data = data[relevant_columns]

        if len(data) < as_samples(self.min_gsd_duration_s, sampling_rate_hz):
            self.gs_list_ = _unify_gs_df(pd.DataFrame(columns=["start", "end"]))
            return self

        # Filter the data
        try:
            filtered_data = self.pre_filter.clone().filter(data, sampling_rate_hz=sampling_rate_hz).transformed_data_
        except ValueError as e:
            if "padlen" in str(e):
                warnings.warn("Data is too short for the filter. Returning empty gait sequence list.", stacklevel=1)
                self.gs_list_ = _unify_gs_df(pd.DataFrame(columns=["start", "end"]))
                return self
            raise e from None

        # Window data and define activity windows
        window_length_samples = round(self.window_length_s * sampling_rate_hz)
        window_overlap_samples = round(window_length_samples * self.window_overlap)
        windowed_filtered_data = sliding_window_view(
            filtered_data["acc_is"].to_numpy(),
            window_length_samples,
            window_overlap_samples,
        )
        windowed_data = sliding_window_view(data["acc_is"].to_numpy(), window_length_samples, window_overlap_samples)

        # Note: We move all the filtering at the beginning here to reduce the number of windows we need to actually
        #       find peaks in later.
        #       In the original implementation only the first step is performed here.
        #       The rest is performed after the step detection.
        # General Activity Recognition
        activity_windows = windowed_filtered_data.std(axis=1, ddof=1) > self.std_activity_threshold
        active_windows_no_grav = windowed_data[activity_windows] - GRAV_MS2
        # Second threshold (not sure why required, but it is in the original implementation)
        activity_windows[activity_windows] &= (
            active_windows_no_grav - active_windows_no_grav.mean(axis=1)[:, None]
        ).mean(axis=1) > self.mean_activity_threshold
        # Standing filtering
        # This should remove windows where the person is not standing upright by checking the value of acc_v
        # Note: We perform the standing filtering here instead of after the step detection
        activity_windows[activity_windows] &= (
            windowed_data[activity_windows].mean(axis=1) > self.acc_v_standing_threshold
        )

        # We shortcut here, if there are no activity windows
        if not activity_windows.any():
            self.gs_list_ = _unify_gs_df(pd.DataFrame(columns=["start", "end"]))
            return self

        # Convolve the data with sin signal
        # The template is equivalent to cycle of a sin wave with a frequency of `sin_template_freq_hz`
        sin_template = np.sin(np.linspace(0, 2 * np.pi, round(sampling_rate_hz / self.sin_template_freq_hz)))
        # We normalize the sin_template with the sampling rate! Otherwise, the convolution amplitude would scale with
        # the sampling rate.
        sin_template /= sampling_rate_hz

        # We split this explicitly by the two axis here, as both axis have slightly different configurations and
        # thresholds
        n_peaks = np.zeros((activity_windows.shape[0], 2), dtype=np.int32)
        # IS:
        # NOTE: THE -GRAV_MS2! I missed that for the longest time in the original implementation.
        # TODO: Explore if using the filtered data here would be better. Note, that substracting GRAV_MS2 does not
        #  make sense then.
        data_channel = data["acc_is"].to_numpy() - GRAV_MS2
        convolved_data = np.convolve(data_channel, sin_template, mode="same")
        convolved_data_windowed = sliding_window_view(convolved_data, window_length_samples, window_overlap_samples)
        n_peaks[activity_windows, 0] = self._find_peaks(
            convolved_data_windowed[activity_windows],
            sampling_rate_hz=sampling_rate_hz,
            step_detection_threshold=self.step_detection_thresholds[0],
            max_allowed_steps_per_s=self.allowed_steps_per_s[1],
            use_original_peak_detection=self.use_original_peak_detection,
        )

        # PA:
        # NOTE: NO -GRAV_MS2!
        data_channel = data["acc_pa"].to_numpy()
        convolved_data = np.convolve(data_channel, sin_template, mode="same")
        convolved_data_windowed = sliding_window_view(convolved_data, window_length_samples, window_overlap_samples)
        n_peaks[activity_windows, 1] = self._find_peaks(
            convolved_data_windowed[activity_windows],
            sampling_rate_hz=sampling_rate_hz,
            step_detection_threshold=self.step_detection_thresholds[1],
            max_allowed_steps_per_s=self.allowed_steps_per_s[1],
            use_original_peak_detection=self.use_original_peak_detection,
        )

        del data_channel, convolved_data, convolved_data_windowed

        # Note: The original implementation uses a different threshold for the second axis.
        #       Namely, on a 3-second window, they expect a minimum of only 1 step instead of 3.
        #       There is no clear reason for this, so I decided to use the same threshold for both axes.
        selected_windows = (n_peaks > self.allowed_steps_per_s[0] * self.window_length_s) & (
            n_peaks <= self.allowed_steps_per_s[1] * self.window_length_s
        )

        # Only keep windows that are selected for both axes
        selected_windows = selected_windows.all(axis=1)

        # For the remaining windows, we calculate the mean of acc_v in the first and the last second to check for
        # changes in the orientation.
        # If the difference is above the threshold, we remove the window
        acc_v_windowed = windowed_data[selected_windows]
        one_second = as_samples(1, sampling_rate_hz)
        first_second_mean = acc_v_windowed[:, :one_second].mean(axis=1)
        last_second_mean = acc_v_windowed[:, -one_second:].mean(axis=1)
        selected_windows[selected_windows] &= (
            np.abs((first_second_mean - last_second_mean) / first_second_mean) <= self.allowed_acc_v_change_per_window
        )

        # Now we turn the selected windows into a list of gait sequences
        selected_windows_index = np.where(selected_windows)[0]
        # We initialize an empty array with two columns, where each row represents a gsd. First column is the start
        # index, second column is the end index.
        gs_list = np.empty((len(selected_windows_index), 2), dtype="int64")
        # We convert the window indices to sample indices of the original data
        gs_list[:, 0] = selected_windows_index * (window_length_samples - window_overlap_samples)
        # We add one to make the end index inclusive
        gs_list[:, 1] = gs_list[:, 0] + window_length_samples + 1

        # Merge overlapping windows
        gs_list = merge_intervals(gs_list)

        gs_list = pd.DataFrame(gs_list, columns=["start", "end"]).clip(0, len(data))

        # Finally, we remove all gsds that are shorter than `min_duration` seconds
        gs_list = gs_list[(gs_list["end"] - gs_list["start"]) / sampling_rate_hz >= self.min_gsd_duration_s]

        self.gs_list_ = _unify_gs_df(gs_list.reset_index(drop=True).copy())

        return self

    def _find_peaks(
        self,
        data_windows: np.ndarray,
        *,
        step_detection_threshold: float,
        sampling_rate_hz: float,
        max_allowed_steps_per_s: float,
        use_original_peak_detection: bool,
    ) -> np.ndarray:
        # Per window (that is considered activity), calculate the number of peaks
        # The number of peaks is expected to be the number of steps
        if use_original_peak_detection:
            return vec_find_n_peaks_original(
                data_windows,
                step_detection_threshold,
                # Note the 0.1 seconds here has to be relatively low, as the meaning of that value is not the
                # distance between peaks, but the distance between consecutive zero-crossings in the signal.
                # This means, that if we want to detect relatively narrow peaks, we need to set this value low.
                as_samples(0.1, sampling_rate_hz),
            )
        return vec_find_n_peaks(
            data_windows,
            step_detection_threshold,
            # For the normal find peak version, we derive the min distance from the allowed steps per second
            as_samples(1 / max_allowed_steps_per_s, sampling_rate_hz),
        )


# Note: I HATE THIS! Implementing a peak detection algorithm by hand feels like a huge stupid antipattern.
#       It makes everything harder to maintain, and is exactly the thing we wanted to avoid with the reimplementation...
#       However, doing it this way results in an up to 2x speedup on long recordings, so I guess it is worth it.
@guvectorize([(float64[:], float32, float32, int32[:])], "(n),(),()->()", target="parallel", cache=True)
def _find_n_peaks_2d(signal: np.ndarray, threshold: float, distance: float, n_peaks: np.ndarray) -> None:
    """Parallelized version of find_n_peaks for processing each row of a 2D array.

    This function is meant to be applied along an axis of a 2D array in parallel.
    """
    # Initialize n_peaks to 0
    n_peaks[0] = 0

    # Note: The original implementation checks "if any sample is larger than the threshold" here.
    #       Our check should be equivalent, as we use the max of the signal.
    if signal.max() < threshold:
        return

    # Step 1: Find all local maxima
    peaks = []
    n = len(signal)

    for i in range(1, n - 1):
        if (signal[i - 1] < signal[i] > signal[i + 1]) and signal[i] >= threshold:
            peaks.append(i)

    # Step 2: Enforce the minimum distance constraint
    if len(peaks) > 1:
        filtered_peaks = [peaks[0]]  # Always take the first peak
        for peak in peaks[1:]:
            if peak - filtered_peaks[-1] >= distance:
                filtered_peaks.append(peak)
        peaks = filtered_peaks

    # Set the output n_peaks to the number of peaks found
    n_peaks[0] = len(peaks)


@guvectorize([(float64[:], float32, float32, int32[:])], "(n),(),()->()", target="parallel", cache=True)
def _find_n_peaks_exact_matlab_replication(  # noqa: C901, PLR0912
    signal: np.ndarray, threshold: float, min_distance: float, peaks_out: np.ndarray
) -> None:
    """Enhanced peak detection algorithm based on zero-crossings and local maxima.

    Args:
        signal: Input signal array
        threshold: Amplitude threshold for peak detection
        min_distance: Minimum distance between peaks
        peaks_out: Output array containing number of peaks found
    """
    # Initialize output
    peaks_out[0] = 0

    # Note: The original implementation checks "if any sample is larger than the threshold" here.
    #       Our check should be equivalent, as we use the max of the signal.
    if signal.max() < threshold:
        return

    # Normalize signal
    normalized_signal = signal - np.mean(signal)

    # Find zero crossings
    # Shift arrays to detect sign changes
    signs = np.sign(normalized_signal)
    zero_crossings = np.where(signs[:-1] != signs[1:])[0]

    if len(zero_crossings) == 0:
        return

    # Find crossing points that meet minimum distance requirement
    crossings_padded = np.zeros(len(zero_crossings) + 1)
    crossings_padded[:-1] = zero_crossings
    valid_distances = (crossings_padded[1:] - zero_crossings) > min_distance
    candidate_points = zero_crossings[valid_distances]

    if len(candidate_points) == 0:
        return

    # Select every other point as potential peak boundary
    boundary_points = candidate_points[::2]

    # Check for gaps between boundary points
    point_distances = np.zeros(len(boundary_points) + 1)
    point_distances[1:] = boundary_points
    interval_sizes = point_distances[1:] - boundary_points

    # Add intermediate points for large gaps
    if len(interval_sizes) > 2:
        mean_interval = np.mean(interval_sizes[1:-1])
        large_gaps = np.where(interval_sizes[1:-1] > 1.2 * mean_interval)[0] + 1

        if len(large_gaps) > 0:
            # Add midpoints for large gaps
            midpoints = boundary_points[large_gaps - 1] + (
                (boundary_points[large_gaps] - boundary_points[large_gaps - 1]) // 2
            )
            interval_boundaries = np.sort(np.concatenate((boundary_points, midpoints)))
        else:
            interval_boundaries = boundary_points
    else:
        interval_boundaries = boundary_points

    num_intervals = len(interval_boundaries)
    if num_intervals == 0:
        return

        # Add endpoint if necessary
    if len(normalized_signal) > interval_boundaries[-1] + 2:
        interval_boundaries = np.append(interval_boundaries, len(normalized_signal))
    else:
        num_intervals -= 1

    if num_intervals == 0:
        return

    # Find peaks between interval boundaries
    peak_locations = []
    for i in range(num_intervals):
        start = interval_boundaries[i] + 1
        end = interval_boundaries[i + 1] - 1

        if end > start:
            segment = normalized_signal[start:end]
            if len(segment) > 0:
                location = start + np.argmax(segment)
                # This step is done outside the find peaks function in the original implementation
                if normalized_signal[location] > threshold:
                    peak_locations.append(location)

    # Set output
    peaks_out[0] = len(peak_locations)


def vec_find_n_peaks(signal: np.ndarray, threshold: float, distance: float) -> np.ndarray:
    """
    Vectorized version of find_n_peaks for processing a 1D array.

    Parameters
    ----------
    - signal: 1D array of signal values.
    - threshold: Minimum height required for a peak.
    - distance: Minimum distance required between peaks.

    Returns
    -------
    - Number of peaks found in the signal.
    """
    output = np.zeros(signal.shape[0], dtype=np.int32)
    _find_n_peaks_2d(signal, threshold, distance, output)
    return output


def vec_find_n_peaks_original(signal: np.ndarray, threshold: float, distance: float) -> np.ndarray:
    """
    Vectorized version of find_n_peaks for processing a 1D array.

    Parameters
    ----------
    - signal: 1D array of signal values.
    - threshold: Minimum height required for a peak.
    - distance: Minimum distance required between peaks.

    Returns
    -------
    - Number of peaks found in the signal.
    """
    output = np.zeros(signal.shape[0], dtype=np.int32)
    _find_n_peaks_exact_matlab_replication(signal, threshold, distance, output)
    return output
