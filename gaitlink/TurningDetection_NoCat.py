from typing import Tuple, List
import numpy as np
from gaitmap.utils.signal_processing import butter_lowpass_filter_1d
from scipy.signal import find_peaks
class TurningDetection_NoCat:

    # Class is a blueprint for creating objects, defining structure and behaviour of objects (or instances).
    # Here, we define 'TurningDetection' class to represent turns, including their attribute (innit) and method of
    # detecting turns. The code then creates instances of 'TurningDetection_NoCat' to represent turns from individual
    # files or scnarios

    """
    performs turning detection according to El-Gohary et al.

      Parameters
      ----------
      fc_hz: cutoff freq for turning detection
      velocity_dps: threshold for turning velocity_dps [deg/s]
      height: minimal height for turning peaks
      concat_length: maximal distance of turning sequences to be concatenated [in s]
      min_duration_seconds: minimal length of a turning sequence [in s]
      max_duration_seconds: maximal length of a turning sequence [in s]
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

    # parameters required for turning detection, which are used to store data nad results within instances
    # These attributes are not initialised in this section
    data_: np.ndarray
    fc_hz: float
    velocity_dps: float
    height: float
    concat_length: float
    min_duration_seconds: float
    max_duration_seconds: float
    angle_threshold_degrees: Tuple[int, int]
    complete_turns_list_: List
    suitable_angles_: List
    Turn_Start_seconds: List
    Turn_End_seconds: List
    all_turns_: List
    all_angles_degrees: List
    duration_list_seconds: List
    duration_list_fps: List

    # Initialises the parameters in this section and is called whenever an instance of a class is created.
    # These provide default values for each of the parameters above, which can be overidden in each instance
    # This allows customization of behaviour of each instance
    # Self refers to a current instance of the class. It's a pointer that allows you access and work with that
    # object (instance)
    def __init__(
        self,
        fc_hz=1.5,
        velocity_dps=5,
        height=15,
        concat_length=0.05,
        min_duration_seconds=0.5,
        max_duration_seconds=10,
        angle_threshold_degrees=(160, 200),
    ):
        self.angle_threshold_degrees = angle_threshold_degrees
        self.concat_length = concat_length
        self.height = height
        self.velocity_dps = velocity_dps
        self.fc_hz = fc_hz
        self.max_duration_seconds = max_duration_seconds
        self.min_duration_seconds = min_duration_seconds

    # Method within the class
    def detect_turning_points(self, data_, fs_hz):
        # Assign data to instance of class
        self.data_ = data_
        # gyr_z_lp = np.abs(lowpass_filtering(self.data_, fs_hz, self.fc_hz))
        gyr_z_lp = butter_lowpass_filter_1d(data=self.data_, sampling_rate_hz=fs_hz, cutoff_freq_hz=self.fc_hz, order=4)
        peaks, _ = find_peaks(gyr_z_lp, height=self.height)
        # compute turn durations
        
        # find indicies of smaller gyr-z values
        lower_idx = np.where(np.array(gyr_z_lp) < self.velocity_dps)[0]
        turning_list = []
        self.duration_list_seconds = []
        self.duration_list_fps =[]

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

            # calculate duration_fps of turn in samples
            duration_fps = lower_idx[right_border_idx] - lower_idx[left_border_idx]
            duration_seconds = duration_fps / 100
            
            # store turn as list [starting index, lenght]
            turning_list.append([lower_idx[left_border_idx], duration_fps])
            self.duration_list_fps.append([duration_fps])
            self.duration_list_seconds.append([duration_seconds])

        # combine start/end point of turns
        combined_turns = []
        if len(turning_list) == 0:
            self.all_angles_degrees, self.all_turns_, self.complete_turns_list_ = [], [], []
            return self

        # calculate endpoints of turns by adding turn duration_fps to start points
        endpoints = np.sum(turning_list, axis=1)
        # extract starting points of turns
        startpoints = np.array(turning_list)[:, 0]

        self.Turn_End_seconds = endpoints / 100
        self.Turn_Start_seconds = startpoints / 100

        for i, label in enumerate(turning_list):

            new_endpoint = endpoints[i]
            # add new turn to list
            combined_turns.append([label[0], new_endpoint])

        # exclude section if length is not suitable
        lengths = np.diff(combined_turns, axis=1) / fs_hz
        suitable_index_list = [
            idx for idx, length in enumerate(lengths) if (self.min_duration_seconds <= length <= self.max_duration_seconds)
        ]
        self.all_turns_ = [combined_turns[t] for t in suitable_index_list]

        # calculate turning angles and eliminate too small angles
        turning_angles_degrees = []
        for t in self.all_turns_:
            # approximate integral by summing up the respective section
            integral = np.sum(gyr_z_lp[t[0] : t[1]] / fs_hz)
            turning_angles_degrees.append(integral)

        self.all_angles_degrees = turning_angles_degrees
        self.complete_turns_list_ = [self.all_turns_]
        return self
