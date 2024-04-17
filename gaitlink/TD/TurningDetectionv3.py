from typing import Tuple, List
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.integrate import cumtrapz
from matplotlib import pyplot as plt
from gaitlink.gaitlink.data_transform._filter import ButterworthFilter

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
    fs_hz: float
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
    duration_list_frames: List
    magnitude_list: List
    startstop_list: List

    def __init__(
        self,
        fc_hz=1.5,
        velocity_dps=5,
        height=15,
        concat_length=0.05,
        min_duration=0.5,
        max_duration=10,
        angle_threshold_degrees=45,
        filter_type='sos'
    ):
        self.Start = None
        self.AngularVelocity_mean = None
        self.AngularVelocity_peak = None
        self.StartStop = None
        self.AngularVelocity_dps = None
        self.turning_list = None
        self.gyr_x_lp = None
        self.gyr_x_lp_abs = None
        self.angle_threshold_degrees = angle_threshold_degrees
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.concat_length = concat_length
        self.height = height
        self.velocity_dps = velocity_dps
        self.cutoff_freq_hz = fc_hz
        self.filter_type = 'lowpass'
        self.order=4
        self.yaw_angle = None
        self.TurnMagnitude = None
        self.TurnDur = None

    def detect_turning_points(self, data_, fs_hz):
        self.data_ = data_
        # Filter gyr_x
        butterworth_filter = ButterworthFilter(order=4, cutoff_freq_hz=1.5, zero_phase=True)
        butterworth_filter.filter(self.data_, sampling_rate_hz=100)
        gyr_x_lp = butterworth_filter.filtered_data_
        gyr_x_lp_abs = abs(butterworth_filter.filtered_data_)
        peaks, _ = find_peaks(gyr_x_lp_abs, height=self.height)

        # Calculate yaw angle
        t = np.linspace(0, (len(gyr_x_lp) - 1) / fs_hz, len(gyr_x_lp))
        yaw_angle = cumtrapz(gyr_x_lp, t, initial=0)
    #    plt.plot(yaw_angle)
    #      plt.xlim(left=0, right=len(yaw_angle))

        # compute turn durations
        # find indicies of smaller gyr-z values
        lower_idx = np.where(np.array(gyr_x_lp_abs) < self.velocity_dps)[0]
        turning_list = []
        self.duration_list_seconds = []
        self.magnitude_list = []

        # Calculate zero crossings
        CrossVal = 5
        zci = lambda v: np.where(v[:-1] * np.roll(v, -1)[:-1] <= 0)[0]
        zero_crossings = zci(gyr_x_lp-CrossVal)

        gyr_x_lp.reset_index(drop=True).plot(y="gyr_x")
        plt.plot(zero_crossings, gyr_x_lp.iloc[zero_crossings], "o", label="ref_end")
        plt.plot(peaks, gyr_x_lp.iloc[peaks], "o", label="ref_end")

        # Identify 5deg/s crossings preceding and following each maximum
        x1 = []
        x2 = []

        for p in peaks:
            bef = zero_crossings < p
            if not np.any(bef):
                x1.append(1)
            else:
                bef = zero_crossings[bef]
                x1.append(bef[-1])

            aft = zero_crossings > p
            if not np.any(aft):
                x2.append(len(gyr_x_lp))
            else:
                aft = zero_crossings[aft]
                x2.append(aft[0])

        x1 = np.array(x1)
        x2 = np.array(x2)
        all_changes = np.unique(np.sort(np.concatenate([x1, x2])))

        # Turn definitions
        StartStop = np.zeros((len(all_changes) - 1, 2))
        TurnM = np.zeros(len(all_changes) - 1)
        TurnDur = np.zeros(len(all_changes) - 1)
        j = 0
        for i in range(1, len(all_changes)):
            start_index = all_changes[i - 1]
            end_index = all_changes[i]

            # Ensure indices are within bounds
            start_index = min(start_index, len(yaw_angle) - 1)
            end_index = min(end_index, len(yaw_angle) - 1)

            StartStop[j, :] = [start_index, end_index]
            TurnM[j] = yaw_angle[end_index] - yaw_angle[start_index]
            TurnDur[j] = (end_index - start_index) / 100  # fs
            j += 1

        self.magnitude_list = TurnM
        self.duration_list_seconds = TurnDur
        self.turning_list = turning_list
        self.startstop_list = StartStop
        self.gyr_x_lp = gyr_x_lp
        self.gyr_x_lp_abs = gyr_x_lp_abs
        self.yaw_angle = yaw_angle

    # def post_process(self, turning_list, fs_hz, gyr_x_lp_abs, gyr_x_lp, TurnDuration, TurnMagnitude, StartStop,
    #                  yaw_angle):
        i = 0
        while i < len(TurnDur) - 1:
            if TurnDur[i] < 0.05 and (np.sign(TurnM[i]) == np.sign(TurnM[i + 1])):
                StartStop[i, :] = [StartStop[i, 0], StartStop[i + 1, 1]]
                StartStop = np.delete(StartStop, i + 1, axis=0)
                TurnM[i] = yaw_angle[StartStop[i, 1]] - yaw_angle[StartStop[i, 0]]
                TurnM = np.delete(TurnM, i + 1)
                TurnDur[i] = (StartStop[i, 1] - StartStop[i, 0]) / fs
                TurnDur = np.delete(TurnDur, i + 1)
            else:
                i += 1

        Check = np.ones_like(TurnDur, dtype=bool)

        for i in range(len(TurnDur)):
            if TurnDur[i] > 10 or TurnDur[i] < 0.5:
                Check[i] = False
            if abs(TurnM[i]) < 45:  # turn threshold
                Check[i] = False

        magnitude_dps = TurnM[Check]
        duration_sec = TurnDur[Check]
        startstop_sec = StartStop[Check, :]
        angular_velocity_dps = gyr_x_lp[int(StartStop[0, 0]):int(StartStop[0, 1]) + 1]
        peak_velocity_dps = max(angular_velocity_dps, key=abs, default=None)
        mean_velocity_dps = np.mean(angular_velocity_dps)


        # Convert all variables to 1-dimensional arrays
        result_dict = {
            'TurnMagnitude_dps': np.array(magnitude_dps).flatten(),
            'TurnDur_sec': np.array(duration_sec).flatten(),
            'Start_sec': np.array(startstop_sec[:, 0]).flatten(),
            'Stop_sec': np.array(startstop_sec[:, 1]).flatten(),
            'AngularVelocity_dps': np.array(angular_velocity_dps).flatten(),
            'PeakVelocity_dps': peak_velocity_dps,
            'MeanVelocity_dps': mean_velocity_dps
        }

        # ...

        detected_start = pd.DataFrame({"td": startstop_sec[:, 0]}).rename_axis(index="td_id")
        detected_end = pd.DataFrame({"td": startstop_sec[:, 1]}).rename_axis(index="td_id")
        turn_duration_sec = pd.DataFrame({"td": duration_sec}).rename_axis(index="td_id")
        angle_deg = pd.DataFrame({"td": magnitude_dps}).rename_axis(index="td_id")

        self.td_list_ = detected_start
        self.end_list_ = detected_end
        self.duration_list_ = turn_duration_sec
        self.angle_list_ = angle_deg

        return self

