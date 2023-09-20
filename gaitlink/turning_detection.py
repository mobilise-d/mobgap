from typing import Tuple, List

import numpy as np
from gaitmap.utils.signal_processing import butter_lowpass_filter_1d
from scipy.signal import find_peaks

class TurningDetection:

    """
    performs turning detection according to El-Gohary et al.

      Parameters
      ----------
      fc: cutoff freq for turning detection
      velocity: threshold for turning velocity [deg/s]
      height: minimal height for turning peaks
      concat_length: maximal distance of turning sequences to be concatenated [in s]
      min_length: minimal length of a turning sequence [in s]
      max_length: maximal length of a turning sequence [in s]
      angle_threshold: tuple with threshold for detected turning angle


      Attributes
      ----------
      data_: gyr_z data
      complete_turns_list_: list of turning intervals with turning angles > angle_threshold,
                         each is represented as [start, length]
      suitable_angles_:  all corresponding turning angles of the turns in complete_turns_list
      all_turns_: all turns and their position within the signal
      all_angles_: corresponding turning angles to all turns
    """

    # parameters required for turning detection
    data_: np.ndarray
    fc: float
    velocity: float
    height: float
    concat_length: float
    min_length: float
    max_length: float
    angle_threshold: Tuple[int, int]
    complete_turns_list_: List
    suitable_angles_: List
    all_turns_: List
    all_angles_: List

    def __init__(
        self,
        fc=1.5,
        velocity=5,
        height=15,
        concat_length=0.05,
        min_length=0.5,
        max_length=10,
        angle_threshold=(160, 200),
    ):
        self.angle_threshold = angle_threshold
        self.max_length = max_length
        self.min_length = min_length
        self.concat_length = concat_length
        self.height = height
        self.velocity = velocity
        self.fc = fc

    def detect_turning_points(self, gyr_z_data, fs):

        self.data_ = gyr_z_data
       # gyr_z_lp = np.abs(lowpass_filtering(self.data_, fs, self.fc))
        gyr_z_lp = butter_lowpass_filter_1d(data=self.data, sampling_rate_hz=fs, cutoff_freq_hz=self.fc, order=4)
        peaks, _ = find_peaks(gyr_z_lp, height=self.height)
        # compute turn durations
        # find indicies of smaller gyr-z values
        lower_idx = np.where(np.array(gyr_z_lp) < self.velocity)[0]
        turning_list = []
        # iterate over gyroscope peaks and determine borders by finding closest points deceeding threshold
        for p in peaks:
            # calculate distances in samples to peak index
            distances = lower_idx - p
            try:
                # idx of left border is the closest point to p on the left, i.e the largest negative distance value
                left_border_idx = list(distances).index(max(distances[distances < 0]))
                # idx of right border is the closest point to p on the right, i.e the smallest positive distance value
                right_border_idx = list(distances).index(min(distances[distances > 0]))
            # in case the peak is a boundary value at the very beginning or end of gait sequence
            # -> no border beneath turn threshold might be found
            except ValueError:
                continue
            # calculate duration of turn in samples
            duration = lower_idx[right_border_idx] - lower_idx[left_border_idx]
            # store turn as list [starting index, lenght]
            turning_list.append([lower_idx[left_border_idx], duration])

        # concatenate turns with less than 0.5 in between if they are facing in the same direction
        concatenated_turns = []
        if len(turning_list) == 0:
            self.all_angles_, self.all_turns_, self.complete_turns_list_ = [], [], []
            return self

        # calculate endpoints of turns
        endpoints = np.sum(turning_list, axis=1)
        # extract starting points of turns
        startpoints = np.array(turning_list)[:, 0]
        # calculate distances between succeeding turns
        diffs = startpoints[1:] - endpoints[:-1]
        # calculate indicies of turns that might belong together
        concat_idx = np.where((diffs <= self.concat_length * fs))[0]

        ctr = 0
        for i, label in enumerate(turning_list):
            # turn has been processed already
            if i < ctr:
                continue
            while ctr in concat_idx:
                # check if turns are facing in the same direction
                # calculate integral values
                first = np.sum(self.data_[startpoints[ctr] : endpoints[ctr]])
                second = np.sum(self.data_[startpoints[ctr + 1] : endpoints[ctr + 1]])
                # check if they have the same sign
                if np.sign(first) == np.sign(second):
                    ctr += 1
                    continue
                break
            # set new endpoint for elongated turn
            new_endpoint = endpoints[ctr]
            # add new turn to list
            concatenated_turns.append([label[0], new_endpoint])
            ctr += 1

        # exclude section if length is not suitable
        lengths = np.diff(concatenated_turns, axis=1) / fs
        suitable_index_list = [
            idx for idx, length in enumerate(lengths) if (self.min_length <= length <= self.max_length)
        ]
        self.all_turns_ = [concatenated_turns[t] for t in suitable_index_list]

        # calculate turning angles and eliminate too small angles
        turning_angles = []
        for t in self.all_turns_:
            # approximate integral by summing up the respective section
            integral = np.sum(gyr_z_lp[t[0] : t[1]] / fs)
            turning_angles.append(integral)
        self.all_angles_ = turning_angles
        # store turns with turning angle within the range of angle threshold
        suitable_angles = np.where(
            (np.array(self.all_angles_) > self.angle_threshold[0])
            & (np.array(self.all_angles_) < self.angle_threshold[1])
        )[0]
        self.complete_turns_list_ = [self.all_turns_[t] for t in suitable_angles]
        self.suitable_angles_ = suitable_angles
        return self
