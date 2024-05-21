import numpy as np  # TODO: check that all packages are actually needed
import pandas as pd
from scipy.signal import butter, lfilter
from typing import Any
from typing_extensions import Self, Unpack
from tpcp import cf

from gaitmap.trajectory_reconstruction.orientation_methods import MadgwickAHRS
from gaitmap.utils.rotations import rotate_dataset_series
from mobgap.sl.base import BaseSlCalculator, base_sl_docfiller
from mobgap.data_transform import HampelFilter
from mobgap.data_transform.base import BaseFilter
from mobgap.utils.conversions import as_samples
from mobgap.utils.interpolation import robust_step_para_to_sec

@base_sl_docfiller
class SlZijlstra(BaseSlCalculator):
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
    8. Remove step length outliers during the gait sequence --> Hampel filter based on median absolute deviation
    9. Interpolation of StepLength values --> Length per second values
    10. Remove length per second outliers during the gait sequence --> Hampel filter based on median absolute deviation


    This is based on the implementation published as part of the mobilised project [3]_.
    However, this implementation deviates from the original implementation in some places.
    For details, see the notes section and the examples.

    Parameters
    ----------

    beta
        The beta value for the (optional) reorientation with Madgwick filter.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(sl_list_)s

    Notes
    -----
    Points of deviation from the original implementation and their reasons:

    -

    .. [1] W. Zijlstra, & A. L. Hof, "Assessment of spatio-temporal gait parameters from trunk accelerations during
        human walking" Gait & posture, vol. 18, no. 2, pp. 1-10, 2003.
    .. [2] A. Soltani, et al. "Algorithms for walking speed estimation using a lower-back-worn inertial sensor:
        A cross-validation on speed ranges." IEEE TNSRE, vol 29, pp. 1955-1964, 2021.
    .. [3] https://github.com/mobilise-d/Mobilise-D-TVS-Recommended-Algorithms/blob/master/SLA_SLB/STRIDELEN.m

    """

    beta: float
    step_length_smoothing: BaseFilter
    max_interpolation_gap_s: int
    def __init__(self, *, beta: float = 0.1,
                 step_length_smoothing: BaseFilter = cf(HampelFilter(2, 3.0)),
                 max_interpolation_gap_s: int = 3) -> None:
        self.beta = beta # TODO: understand how to pass beta and rotation method
        self.step_length_smoothing = step_length_smoothing  # TODO: understand how to pass step_length_smoothing to _stride2sec
        self.max_interpolation_gap_s = max_interpolation_gap_s
    @base_sl_docfiller
    def calculate(self,
                  data: pd.DataFrame, *,
                  initial_contacts: pd.Series,
                  sampling_rate_hz: float,
                  sensor_height: float,
                  tuning_coefficient: float,
                  align_flag: bool = False, **_: Unpack[dict[str, Any]] # TODO: which default value for orientation method?
                  ) -> Self:
        """%(calculate_short)s.

        Parameters
        ----------
        %(calculate_para)s

        %(calculate_return)s

        """
        self.data = data
        self.initial_contacts = initial_contacts
        self.sampling_rate_hz = sampling_rate_hz
        self.sensor_height = sensor_height
        self.tuning_coefficient = tuning_coefficient # TODO: Should the tuning coefficient be user selectable?
        self.aligned = align_flag
        # 1. Sensor alignment (optional): Madgwick complementary filter
        newAcc = data[['acc_x', 'acc_y', 'acc_z']]  # consider acceleration
        if align_flag:  # TODO: how to make rotation optional?
            orientation_method = MadgwickAHRS(beta = 0.1)
            # perform rotation
            rotated_data = rotate_dataset_series(data, orientation_method.estimate(data, sampling_rate_hz=sampling_rate_hz).orientation_object_[:-1])
            newAcc = rotated_data[['acc_x', 'acc_y', 'acc_z']]  # consider acceleration

        # 2. High-pass filtering --> lower cut-off: 0.1 Hz, filter design: Butterworth IIR, filter order: 4
        vacc = newAcc['acc_x']  # TODO: check that acceleration is passed to the calculate method in m/s^2
        fc = 0.1  # cut-off frequency
        [df_, cf_] = butter(4, fc / (sampling_rate_hz / 2), 'high')  # design high-pass filter
        vacc_high = lfilter(df_, cf_, vacc)  # filtering

        # 3. Integration of vertical acceleration --> vertical speed
        # 4. Drift removal (high-pass filtering) --> lower cut-off: 1 Hz, filter design: Butterworth IIR, filter order: 4
        # 5. Integration of vertical speed --> vertical displacement d(t)
        # 6. Compute total vertical displacement during the step (d_step):
        # d_step = |max(d(t)) - min(d(t))|
        # 7. Biomechanical model:
        # StepLength = A * 2 * sqrt(2 * LBh * d_step - d_step^2)
        # A: tuning coefficient, optimized by training for each population
        # LBh: sensor height in meters, representative of the height of the center of mass
        # TODO: Which is the correct way to pass tuning_coefficient, initial_contacts, sensor_height?
        initial_contacts = np.squeeze(initial_contacts.astype(int))
        sl_zjilstra_v3 = _zjilsV3(vacc_high,
                                  sampling_rate_hz,
                                  tuning_coefficient,
                                  initial_contacts,
                                  sensor_height)

        # 8. Remove step length outliers during the gait sequence --> Hampel filter based on median absolute deviation
        # 9. Interpolation of StepLength values --> Length per second values
        # 10. Remove length per second outliers during the gait sequence --> Hampel filter based on median absolute deviation
        duration = int(np.floor((len(vacc) / sampling_rate_hz)))  # bottom-rounded duration (s)
        ICtime = initial_contacts / sampling_rate_hz  # initial contacts: samples -> sec
        sec_centers = np.arange(0, duration) + 1 # TODO: +1 is to stick to the original implementation
        # padding last value of the calculated step length values so that sl_zjilstra_v3 is as long as ICtime
        if len(sl_zjilstra_v3) < len(ICtime):
            sl_zjilstra_v3 = np.concatenate([sl_zjilstra_v3, np.tile(sl_zjilstra_v3[-1], len(ICtime) - len(sl_zjilstra_v3))])

        slSec_zjilstra_v3 = robust_step_para_to_sec(ICtime,
                                                    sl_zjilstra_v3,
                                                    sec_centers,
                                                    self.max_interpolation_gap_s,
                                                    self.step_length_smoothing.clone())

        slSec = slSec_zjilstra_v3[0:duration]
        self.sl_list_ = pd.DataFrame({'length_m':slSec},
                                     index = as_samples(sec_centers, sampling_rate_hz)
                                     ).rename_axis(index="sec_center_samples")

        return self


def _zjilsV3(vacc_high: np.ndarray, sampling_rate_hz: float, tuning_coefficient: float, initial_contacts: pd.Series, sensor_height: float) -> np.ndarray:
    """
    Step length estimation using the biomechanical model propose by Zijlstra & Hof [1]
    [1] Zijlstra, W., & Hof, A. L. (2003). Assessment of spatio-temporal gait parameters from trunk accelerations
    during human walking. Gait & posture, 18(2), 1-10.
    Inputs:
    - vacc_high: vertical acceleration recorded on lower back, high-pass filtered.
    - sampling_rate_hz: sampling frequency of the acc signal.
    - tuning_coefficient: correction factor estimated by data from various clinical populations
    - initial_contacts: vector containing the timing of initial contacts events (samples)
    - sensor_height: Low Back height, i.e., the distance from ground to sensor location on lower back (in m)

    Output:
    - sl_zjilstra_v3: estimated step length.
    """
    # estimate vertical speed
    vspeed = -np.cumsum(vacc_high) / sampling_rate_hz
    # drift removal (high pass filtering)
    fc = 1
    b, a = butter(4, Wn = fc / (sampling_rate_hz / 2), btype = 'high', output = 'ba') # TODO: why this causes a warning?
    speed_high = lfilter(b, a, vspeed)
    # estimate vertical displacement
    vdis_high_v2 = np.cumsum(speed_high) / sampling_rate_hz
    # initialize the output array as an array of zeros having one less element
    # than the array of ICs
    h_jilstra_v3 = np.zeros(len(initial_contacts) - 1)
    # initial estimates of the stride length
    for k in range(len(initial_contacts) - 1):
        # the k-th stride length value is initially estimated as the absolute difference between the maximum and the
        # minimum (range) of the vertical displacement between the k-th and (k+1)-th IC
        h_jilstra_v3[k] = np.abs(max(vdis_high_v2[initial_contacts[k]:initial_contacts[k + 1]]) -
                                  min(vdis_high_v2[initial_contacts[k]:initial_contacts[k + 1]]))
    # biomechanical model formula
    sl_zjilstra_v3 = tuning_coefficient * np.sqrt(np.abs((2 * sensor_height * h_jilstra_v3) - (h_jilstra_v3 ** 2)))
    return sl_zjilstra_v3