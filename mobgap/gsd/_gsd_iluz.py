from typing import Any

import numpy as np
import pandas as pd
from gaitmap.utils.array_handling import merge_intervals
from scipy.signal import find_peaks
from tpcp import cf
from typing_extensions import Self, Unpack

from mobgap.consts import GRAV_MS2
from mobgap.data_transform import FirFilter
from mobgap.data_transform.base import BaseFilter
from mobgap.gsd.base import BaseGsDetector, base_gsd_docfiller
from mobgap.utils.array_handling import sliding_window_view


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

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(gs_list_)s

    Notes
    -----
    Points of deviation from the original implementation and their reasons:

    - The order of processing is changed.
      In the original implementation, steps are detected early on in the pipeline and later further thresholds on the
      raw signal are used to discard certain parts of the signal and non-gait.
      We flip the order to reduce the number of windows we need to apply step detection to.
      This is done, because the step detection process is the most expensive part of the algorithm.
    - Instead of a custom peak detection algorithm, we use the scipy implementation (:func:`~scipy.signal.find_peaks`).
      This method produces similar but different results.
      Most notably, it does not have a maximal distance parameter.
      However, based on some testing, this parameter did not seem to have a big impact on the results, anyway.
      Overall, the scipy implementation seems to be more robust and detects less false positives
      (i.e. less peaks overall)
    - As the new find-peaks approach finds fewer peaks, we also change the threshold for the number of peaks per window.
      The original implementation expects 3 peaks per 3-second window.
      We use 0.5 steps per second as the lower bound, which means a minimum of 1.5/2 steps per 3-second window.
    - Similarly, the original implementation uses different thresholds for the two signal axis.
      I.e. different numbers of peaks are expected for the two axes.
      As this is not mentioned anywhere in the paper, and using the same threshold for both axis did not seem to have a
      negative impact on the results, we decided to use the same threshold for both axes to simplify the algorithm.
    - All parameters and thresholds are converted the units used in mobgap.
      Specifically, we use m/s^2 instead of g.
    - The original implementation used a check, that if the sum of the signal in the window is below the min-height
      threshold no peaks are detected.
      We assume that this is an error and use the max of the signal instead.

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

    def __init__(
        self,
        *,
        pre_filter: BaseFilter = cf(FirFilter(order=200, cutoff_freq_hz=(0.5, 3), filter_type="bandpass")),
        window_length_s: float = 3,
        window_overlap: float = 0.5,
        std_activity_threshold: float = 0.01 * GRAV_MS2,
        mean_activity_threshold: float = -0.1 * GRAV_MS2,
        acc_v_standing_threshold: float = 0.5 * GRAV_MS2,
        step_detection_thresholds: tuple[float, float] = (
            0.4 * GRAV_MS2,
            1.5 * GRAV_MS2,
        ),
        sin_template_freq_hz: float = 2,
        # Note: The original implementation uses 1 step per second as the lower bound. This means a minimum of 3
        #       steps per 3-second window. We use 0.5 steps per second as the lower bound, which means a minimum of
        #       1.5/2 steps per 3-second window.
        allowed_steps_per_s: tuple[float, float] = (0.5, 3),
        allowed_acc_v_change_per_window: float = 0.15,
        min_gsd_duration_s: float = 5,
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

    @base_gsd_docfiller
    def detect(
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
        # TODO: Add docstring
        relevant_columns = ["acc_x", "acc_z"]
        data = data[relevant_columns]

        # Filter the data
        filtered_data = self.pre_filter.clone().filter(data, sampling_rate_hz=sampling_rate_hz).transformed_data_

        # Window data and define activity windows
        window_length_samples = round(self.window_length_s * sampling_rate_hz)
        window_overlap_samples = round(window_length_samples * self.window_overlap)
        windowed_filtered_data = sliding_window_view(
            filtered_data["acc_x"].to_numpy(),
            window_length_samples,
            window_overlap_samples,
        )
        windowed_data = sliding_window_view(data["acc_x"].to_numpy(), window_length_samples, window_overlap_samples)

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
            self.gs_list_ = pd.DataFrame(columns=["start", "end"])
            return self

        # Convolve the data with sin signal
        # The template is equivalent to cycle of a sin wave with a frequency of `sin_template_freq_hz`
        sin_template = np.sin(np.linspace(0, 2 * np.pi, round(sampling_rate_hz / self.sin_template_freq_hz)))

        def find_n_peaks(signal: np.ndarray, threshold: float) -> int:
            # Note: This is different to the original implementation. Here the sum of the signal and not the max of the
            #       signal is checked. However, I believe that is an error.
            if signal.max() < threshold:
                return 0
            peaks, _ = find_peaks(
                signal,
                # TODO: I don't really know, why a min distance of 1/10 of the sampling rate is used
                # Note: The original find peaks implementation also uses some upper bound. However, due to the way this
                #       is implemented, I don't think it does what is was originally intended to do.
                #       So, I decided to not implement it here in favor of being able to use the basic find_peaks
                #       method.
                distance=sampling_rate_hz / 10,
                height=threshold,
            )
            return len(peaks)

        vec_find_n_peaks = np.vectorize(find_n_peaks, signature="(n), ()->()")

        selected_windows = np.empty((activity_windows.shape[0], 2), dtype=bool)

        for i, axis in enumerate(["acc_x", "acc_z"]):
            convolved_data = np.convolve(data[axis].to_numpy(), sin_template, mode="same")
            convolved_data_windowed = sliding_window_view(convolved_data, window_length_samples, window_overlap_samples)

            # Per window (that is considered activity), calculate the number of peaks
            # The number of peaks is expected to be the number of steps
            n_peaks = np.full(activity_windows.shape, np.nan)
            n_peaks[activity_windows] = vec_find_n_peaks(
                convolved_data_windowed[activity_windows],
                self.step_detection_thresholds[i],
            )
            # Note: The original implementation uses a different threshold for the second axis.
            #       Namely, on a 3-second window, they expect a minimum of only 1 step instead of 3.
            #       There is no clear reason for this, so I decided to use the same threshold for both axes.
            selected_windows[:, i] = (n_peaks >= self.allowed_steps_per_s[0] * self.window_length_s) & (
                n_peaks <= self.allowed_steps_per_s[1] * self.window_length_s
            )

        # Only keep windows that are selected for both axes
        selected_windows = selected_windows.all(axis=1)

        # For the remaining windows, we calculate the mean of acc_v in the first and the last second to check for
        # changes in the orientation.
        # If the difference is above the threshold, we remove the window
        acc_v_windowed = windowed_data[selected_windows]
        first_second_mean = acc_v_windowed[:, : round(sampling_rate_hz)].mean(axis=1)
        last_second_mean = acc_v_windowed[:, -round(sampling_rate_hz) :].mean(axis=1)
        selected_windows[selected_windows] &= (
            np.abs((first_second_mean - last_second_mean) / first_second_mean) <= self.allowed_acc_v_change_per_window
        )

        # Now we turn the selected windows into a list of gait sequences
        selected_windows_index = np.where(selected_windows)[0]
        # We initialize an empty array with two columns, where each row represents a gsd. First column is the start
        # index, second column is the end index.
        gs_list = np.empty((len(selected_windows_index), 2), dtype=int)
        # We convert the window indices to sample indices of the original data
        gs_list[:, 0] = selected_windows_index * (window_length_samples - window_overlap_samples)
        # We add one to make the end index inclusive
        gs_list[:, 1] = gs_list[:, 0] + window_length_samples + 1

        # Merge overlapping windows
        gs_list = merge_intervals(gs_list)

        gs_list = pd.DataFrame(gs_list, columns=["start", "end"])

        # To make sure that the end is inclusive, we add 1 to the end index
        gs_list["end"] += 1

        # Finally, we remove all gsds that are shorter than `min_duration` seconds
        gs_list = gs_list[(gs_list["end"] - gs_list["start"]) / sampling_rate_hz >= self.min_gsd_duration_s]

        self.gs_list_ = gs_list.copy()

        return self
