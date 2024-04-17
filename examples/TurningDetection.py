 from typing import Tuple, List
import numpy as np
from gaitmap.utils.signal_processing import butter_lowpass_filter_1d
from scipy.signal import find_peaks
class TurningDetection:

    """
    performs turning detection according to El-Gohary et al.

      Parameters
      ----------
      fc_hz: cutoff freq for turning detection
      velocity_dps: threshold for turning velocity_dps [deg/s]
      height: minimal height for turning peaks
      concat_length: maximal distance of turning sequences to be concatenated [in s]
      min_duration: minimal length of a turning sequence [in s]
      max_duration: maximal length of a turning sequence [in s]
      angle_threshold_degrees: tuple with threshold for detected turning angle


      Attributes
      ----------
      data_: gyr_z data
      complete_turns_list_: list of turning intervals with turning angles > angle_threshold_degrees,
                         each is represented as [start, length]
      suitable_angles_:  all corresponding turning angles of the turns in complete_turns_list
      all_turns_: all turns and their position within the signal
      all_angles_degrees: corresponding turning angles to all turns
    """

    # parameters required for turning detection
    data_: np.ndarray
    fc_hz: float
    velocity_dps: float
    height: float
    concat_length: float
    min_duration: float
    max_duration: float
    angle_threshold_degrees: Tuple[int, int]
    complete_turns_list_: List
    suitable_angles_: List
    all_turns_: List
    all_angles_degrees: List
    Turn_End_seconds: List
    Turn_Start_seconds: List
    duration_list_seconds: List
    duration_list_frames: List

    def __init__(
        self,
        fc_hz=1.5,
        velocity_dps=5,
        height=15,
        concat_length=0.05,
        min_duration=0.5,
        max_duration=10,
        angle_threshold_degrees=45,
    ):
        self.turning_list = None
        self.gyr_z_lp = None
        self.angle_threshold_degrees = angle_threshold_degrees
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.concat_length = concat_length
        self.height = height
        self.velocity_dps = velocity_dps
        self.fc_hz = fc_hz

    def detect_turning_points(self, data_, fs_hz):
        self.data_ = data_
        gyr_z_lp = butter_lowpass_filter_1d(data=self.data_, sampling_rate_hz=fs_hz, cutoff_freq_hz=self.fc_hz, order=4)
        peaks, _ = find_peaks(gyr_z_lp, height=self.height)
        # compute turn durations
        # find indicies of smaller gyr-z values
        lower_idx = np.where(np.array(gyr_z_lp) < self.velocity_dps)[0]
        turning_list = []
        self.duration_list_seconds = []
        self.duration_list_frames = []
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
            # calculate duration_frames of turn in samples
            duration_frames = lower_idx[right_border_idx] - lower_idx[left_border_idx]
            duration_seconds = duration_frames / fs_hz
            # store turn as list [starting index, lenght]
            turning_list.append([lower_idx[left_border_idx], duration_frames])
            self.duration_list_frames.append([duration_frames])
            self.duration_list_seconds.append([duration_seconds])
        self.turning_list = turning_list
        self.gyr_z_lp = gyr_z_lp

    def post_process(self, turning_list, fs_hz, gyr_z_lp):
        # concatenate turns with less than 0.5 in between if they are facing in the same direction
        concatenated_turns = []
        if len(turning_list) == 0:
            self.all_angles_degrees, self.all_turns_, self.complete_turns_list_ = [], [], []
            return self

        # calculate endpoints_frames of turns
        endpoints_frames = np.sum(turning_list, axis=1)
        # extract starting points of turns
        startpoints_frames = np.array(turning_list)[:, 0]
        # calculate distances between succeeding turns
        diffs = startpoints_frames[1:] - endpoints_frames[:-1]
        # calculate indicies of turns that might belong together
        concat_idx = np.where((diffs <= self.concat_length * fs_hz))[0]
        
        self.Turn_End_seconds = endpoints_frames / fs_hz
        self.Turn_Start_seconds = startpoints_frames / fs_hz

        ctr = 0
        for i, label in enumerate(turning_list):
            # turn has been processed already
            if i < ctr:
                continue
            while ctr in concat_idx:
                # check if turns are facing in the same direction
                # calculate integral values
                first = np.sum(self.data_[startpoints_frames[ctr] : endpoints_frames[ctr]])
                second = np.sum(self.data_[startpoints_frames[ctr + 1] : endpoints_frames[ctr + 1]])
                # check if they have the same sign
                if np.sign(first) == np.sign(second):
                    ctr += 1
                    continue
                break
            # set new endpoint for elongated turn
            new_endpoint = endpoints_frames[ctr]
            # add new turn to list
            concatenated_turns.append([label[0], new_endpoint])
            ctr += 1

        # exclude section if length is not suitable
        lengths = np.diff(concatenated_turns, axis=1) / fs_hz
        suitable_index_list = [
            idx for idx, length in enumerate(lengths) if (self.min_duration <= length <= self.max_duration)
        ]
        self.all_turns_ = [concatenated_turns[t] for t in suitable_index_list]

        # calculate turning angles and eliminate too small angles
        turning_angles = []
        for t in self.all_turns_:
            # approximate integral by summing up the respective section
            integral = np.sum(gyr_z_lp[t[0] : t[1]] / fs_hz)
            turning_angles.append(integral)
        self.all_angles_degrees = turning_angles
        # store turns with turning angle within the range of angle threshold
        suitable_angles = np.where(
            (np.array(self.all_angles_degrees) > self.angle_threshold_degrees)
        )[0]
        self.complete_turns_list_ = [self.all_turns_[t] for t in suitable_angles]
        self.suitable_angles_ = suitable_angles

