from typing import Optional

import numpy as np
import pandas as pd
from gaitmap.utils.array_handling import merge_intervals
from scipy.signal import find_peaks
from typing_extensions import Self

from gaitlink.consts import GRAV_MS2
from gaitlink.data_transform.base import BaseFilter
from gaitlink.data_transform import FirFilter
from gaitlink.gsd.base import BaseGsdDetector
from gaitlink.utils.array_handling import sliding_window_view


class GsdIluz(BaseGsdDetector):
    def __init__(
        self,
        *,
        pre_filter: Optional[BaseFilter] = FirFilter(order=200, cutoff_freq_hz=(0.5, 3), filter_type="bandpass"),
        window_length_s: float = 3,
        window_overlap: float = 0.5,
        std_activity_threshold: float = 0.01 * GRAV_MS2,
        mean_activity_threshold: float = -0.1 * GRAV_MS2,
        step_detection_thresholds: tuple[float, float] = (0.4 * GRAV_MS2, 1.5 * GRAV_MS2),
        acc_v_standing_threshold: float = 0.5 * GRAV_MS2,
        sin_template_freq_hz: float = 2,
        # Note: The original implementation uses 1 step per second as the lower bound. This means a minimum of 3
        #       steps per 3-second window. We use 0.5 steps per second as the lower bound, which means a minimum of
        #       1.5/2 steps per 3-second window.
        allowed_steps_per_s: tuple[float, float] = (0.5, 3),
        allowed_acc_v_change_per_window_percent: float = 15,
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
        self.allowed_acc_v_change_per_window_percent = allowed_acc_v_change_per_window_percent
        self.min_gsd_duration_s = min_gsd_duration_s

    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_) -> Self:
        relevant_columns = ["acc_x", "acc_z"]
        data = data[relevant_columns]

        # Filter the data
        filtered_data = self.pre_filter.clone().filter(data, sampling_rate_hz=sampling_rate_hz).transformed_data_

        # Window data and define activity windows
        window_length_samples = round(self.window_length_s * sampling_rate_hz)
        window_overlap_samples = round(window_length_samples * self.window_overlap)
        windowed_filtered_data = sliding_window_view(
            filtered_data["acc_x"].to_numpy(), window_length_samples, window_overlap_samples
        )
        windowed_data = sliding_window_view(data["acc_x"].to_numpy(), window_length_samples, window_overlap_samples)

        # Note: We move all the filtering at the beginning here to reduce the number of windows we need to actually
        #       find peaks in later.
        #       In the original implementation only the first step is performed here.
        #       The rest is performed after the step detection.
        # General Activity Recognition
        activity_windows = windowed_filtered_data.std(axis=1, ddof=1) > self.std_activity_threshold
        # Basic threshold on normalised data (not sure why this is done, but it is in the original implementation)
        active_windows_no_grav = windowed_data[activity_windows] - GRAV_MS2
        # Standing filtering
        # This should remove windows where the person is not standing upright by checking the value of acc_v
        activity_windows[activity_windows] &= (
            active_windows_no_grav - active_windows_no_grav.mean(axis=1)[:, None]
        ).mean(axis=1) > self.mean_activity_threshold
        # Note: We perform the standing filtering here instead of after the step detection
        activity_windows[activity_windows] &= (
            windowed_data[activity_windows].mean(axis=1) > self.acc_v_standing_threshold
        )

        # Convolve the data with sin signal
        # The template is equivalent to cycle of a sin wave with a frequency of `sin_template_freq_hz`
        sin_template = np.sin(np.linspace(0, 2 * np.pi, round(sampling_rate_hz / self.sin_template_freq_hz)))

        def find_n_peaks(signal: np.ndarray, threshold: float):
            # Note: This is different to the original implementation. Here the sum of the signal and not the max of the
            #       signal is checked. However, I believe that is an error.
            if signal.max() < threshold:
                return 0
            peaks, _ = find_peaks(
                signal,
                # TODO: I don't really know, why a min distance of 1/10 of the sampling rate is used
                # Note: The original find peaks implementation also uses some upper bound. However, due to the way this
                #       is implemented, I don't think it does what is was originally intended to do.
                #       So, I decided to not implement it here in favor of beeing able to use the basic find_peaks
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
                convolved_data_windowed[activity_windows], self.step_detection_thresholds[i]
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
            np.abs((first_second_mean - last_second_mean) / first_second_mean)
            <= self.allowed_acc_v_change_per_window_percent / 100
        )

        # Now we turn the selected windows into a list of gsd
        selected_windows_index = np.where(selected_windows)[0]
        gsd_list = np.empty((len(selected_windows_index), 2), dtype=int)
        gsd_list[:, 0] = selected_windows_index * (window_length_samples - window_overlap_samples)
        # We add one to make the end index inclusive
        gsd_list[:, 1] = gsd_list[:, 0] + window_length_samples + 1

        # Merge overlapping windows
        gsd_list = merge_intervals(gsd_list)

        gsd_list = pd.DataFrame(gsd_list, columns=["start", "end"])

        # Finally, we remove all gsds that are shorter than `min_duration` seconds
        self.gsd_list_ = gsd_list[(gsd_list["end"] - gsd_list["start"]) / sampling_rate_hz >= self.min_gsd_duration_s]

        return self
