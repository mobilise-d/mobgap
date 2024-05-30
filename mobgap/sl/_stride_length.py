from typing import Any

import numpy as np
import pandas as pd
from gaitmap.trajectory_reconstruction.orientation_methods import MadgwickAHRS
from gaitmap.utils.rotations import rotate_dataset_series
from tpcp import cf
from tpcp.misc import set_defaults
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc
from mobgap.data_transform import ButterworthFilter, HampelFilter
from mobgap.data_transform.base import BaseFilter
from mobgap.sl.base import BaseSlCalculator, base_sl_docfiller
from mobgap.utils.conversions import as_samples
from mobgap.utils.interpolation import robust_step_para_to_sec

sl_docfiller = make_filldoc(
    {
        **base_sl_docfiller._dict,
        "sl_short": """
    This uses a robust outlier removal approach to deal with missing initial contacts.
    The output stride length is reported as the average for each 1 second bin within the data.
    An incomplete second at the end is removed.

    Regions (i.e. second bouts) with no initial contacts are interpolated linearly based on the surrounding values, if
    the gap is smaller than the specified maximum interpolation gap.
    Regions without initial contacts that are larger than the specified maximum interpolation gap or at the very start
    or end of the recording are filled with NaNs.

    For more details see the Notes section.
    """,
        "sl_notes": """
    The full process of calculating the stride length per second from initial contacts is as follows:

    1. We calculate the step length from the initial contacts.
       This results in one less step time than initial contacts.
       We replicate the last step length to get the same number of step lengths as initial contacts, to also provide a
       step length value for the last initial contact for the last part of a gait sequence.
    2. We smooth the step length to remove outliers (using a Hampel filter by default).
    3. We calculate the step length per second by averaging the step length over the second.
       This is a little tricky, as some seconds might not contain any initial contacts.
       For all seconds that contain at least one initial contact, we calculate the average step length over the second.
       The step length for seconds without initial contacts is interpolated linearly based on the surrounding values.
       If the gap is larger than the specified maximum interpolation gap, the second is filled with NaNs.
       We don't fill "gaps" at the start and end of the recording, as we assume that gait sequences are cut anyway
       to start and end with a valid initial contact.
    4. We smooth the step length per second again to remove outliers on a per second level.
    5. We multiply by 2 to convert from step length per second to stride length per second

    In case there are less initial contacts than the window size of the smoothing filter, we return NaNs for all
    seconds.

    This approach deviates from the original Matlab implementations in a couple of ways:

    1. The original Matlab implementation has no concept of "maximum interpolation gap".
       Values are interpolated by taking the average of the surrounding values, independent of the length of the gap.
       We decided to introduce a maximum interpolation gap to not mask "issues" in the ic-detection.
    2. When values are interpolated, the original Matlab implementation uses the average of the surrounding values.
       We decided to use linear interpolation instead, as this is more robust to outliers.
       These approaches are identical, if we only need to interpolate a single value, but for larger gaps, linear
       interpolation will start favoring the "closer" neighboring values, which we think is more appropriate.
    """,
        "sl_common_paras": """
    orientation_method
        The orientation method for aligning the sensor axes with the global reference system. If this argument is not passed, 
        orientation is not performed (default value: None).  
        
    step_length_smoothing
        The filter used to smooth the step length.
        This is used to remove outliers in the step length (e.g. when initial contacts are not detected).
        The filter is applied twice, once to the raw step length and a second time on the interpolated step length values
        per second.
        We recommend to use a Hampel filter for this.
    max_interpolation_gap_s
        The maximum gap in seconds that is interpolated.
        If the gap is larger than this value, the second is filled with NaNs.
        We don't fill "gaps" at the start and end of the recording, as we assume that gait sequences are cut anyway
        to start and end with a valid initial contact.
    """,
    },
    doc_summary="Decorator to fill the explanation of the stride length per second interpolation.",
)


@sl_docfiller
class SlZijlstra(BaseSlCalculator):
    """Implementation of the stride length algorithm by Zijlstra and Hof (2003) [1]_ modified by Soltani et al. (2021) [2]_.

    The algorithm includes the following stages starting from vertical acceleration
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
        A: tuning coefficient, optimized by training for each population.
        LBh: sensor height in meters, representative of the height of the center of mass.
    8. Remove step length outliers during the gait sequence --> Hampel filter based on median absolute deviation.
    9. Interpolation of StepLength values --> Step length per second values.
    10. Remove length per second outliers during the gait sequence --> Hampel filter based on median absolute deviation.
    11. Approximated stride length per second values = 2* step length per second values


    This is based on the implementation published as part of the mobilised project [3]_.
    However, this implementation deviates from the original implementation in some places.
    For details, see the notes section and the examples.

    %(sl_short)s

    Parameters
    ----------
    %(sl_common_paras)s

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(stride_length_per_sec_list_)s
    step_length_list_
        Secondary output.
        It provides a Numpy array that contains the raw step length values. The unit is ``m``.
    step_length_per_sec_list_
        Secondary output.
        It provides a Numpy array that contains the interpolated step length values per second. The unit is ``m``.

    Notes
    -----
    %(sl_notes)s

    -

    .. [1] W. Zijlstra, & A. L. Hof, "Assessment of spatio-temporal gait parameters from trunk accelerations during
        human walking" Gait & posture, vol. 18, no. 2, pp. 1-10, 2003.
    .. [2] A. Soltani, et al. "Algorithms for walking speed estimation using a lower-back-worn inertial sensor:
        A cross-validation on speed ranges." IEEE TNSRE, vol 29, pp. 1955-1964, 2021.
    .. [3] https://github.com/mobilise-d/Mobilise-D-TVS-Recommended-Algorithms/blob/master/SLA_SLB/STRIDELEN.m

    """

    orientation_method: MadgwickAHRS
    step_length_smoothing: BaseFilter
    max_interpolation_gap_s: int

    step_length_list_: np.ndarray
    step_length_per_sec_list_: np.ndarray

    class PredefinedParameters:
        """
        The step length scaling factor to be used in the biomechanical model.
        The reported parameters were previously accessed with model.K.ziljsV3.
        Coefficients are divided by four to separate the coefficient from
        the conversion from step length to stride length and from the x2 factor of the formula declared
        in [1, 2].
        """

        step_length_scaling_factor_MS_MS = {"step_length_scaling_factor": 4.587 / 4}
        step_length_scaling_factor_MS_ALL = {"step_length_scaling_factor": 4.739 / 4}

    @set_defaults(**PredefinedParameters.step_length_scaling_factor_MS_MS)
    def __init__(
        self,
        *,
        orientation_method: MadgwickAHRS = None,
        step_length_smoothing: BaseFilter = cf(HampelFilter(2, 3.0)),
        max_interpolation_gap_s: int = 3,
        step_length_scaling_factor: float,
    ) -> None:
        self.step_length_smoothing = (
            step_length_smoothing  # TODO: understand how to pass step_length_smoothing to _stride2sec
        )
        self.max_interpolation_gap_s = max_interpolation_gap_s
        self.orientation_method = orientation_method  # TODO: understand how to optionally perform rotation
        self.step_length_scaling_factor = step_length_scaling_factor

    @sl_docfiller
    def calculate(
        self,
        data: pd.DataFrame,
        *,
        initial_contacts: pd.DataFrame,
        sampling_rate_hz: float,
        sensor_height_m: float,
        **_: Unpack[dict[str, Any]],  # TODO: which default value for orientation method?
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
        self.sensor_height_m = sensor_height_m

        # 1. Sensor alignment (optional): Madgwick complementary filter
        newAcc = data[["acc_x", "acc_y", "acc_z"]]  # consider acceleration
        if self.orientation_method != None:  # TODO: how to make rotation optional?
            # perform rotation
            rotated_data = rotate_dataset_series(
                data, self.orientation_method.estimate(data, sampling_rate_hz=sampling_rate_hz).orientation_object_[:-1]
            )
            newAcc = rotated_data[["acc_x", "acc_y", "acc_z"]]  # consider acceleration
        # 2. High-pass filtering --> lower cut-off: 0.1 Hz, filter design: Butterworth IIR, filter order: 4
        vacc = newAcc["acc_x"]  # TODO: check that acceleration is passed to the calculate method in m/s^2
        HP_filter = ButterworthFilter(order=4, cutoff_freq_hz=0.1, filter_type="highpass")
        vacc_high = HP_filter.filter(vacc, sampling_rate_hz=sampling_rate_hz).filtered_data_
        # 3. Integration of vertical acceleration --> vertical speed
        # 4. Drift removal (high-pass filtering) --> lower cut-off: 1 Hz, filter design: Butterworth IIR, filter order: 4
        # 5. Integration of vertical speed --> vertical displacement d(t)
        # 6. Compute total vertical displacement during the step (d_step):
        # d_step = |max(d(t)) - min(d(t))|
        # 7. Biomechanical model:
        # StepLength = A * 2 * sqrt(2 * LBh * d_step - d_step^2)
        # A: step length scaling factor, optimized by training for each population
        # LBh: sensor height in meters, representative of the height of the center of mass
        # TODO: Which is the correct way to pass step_length_scaling_factor, initial_contacts, sensor_height?
        initial_contacts = np.squeeze(initial_contacts.astype(int))
        sl_zjilstra_v3 = _zjilsV3(
            vacc_high, sampling_rate_hz, self.step_length_scaling_factor, initial_contacts, sensor_height_m
        )

        # 8. Remove step length outliers during the gait sequence --> Hampel filter based on median absolute deviation
        # 9. Interpolation of StepLength values --> Step length per second values
        # 10. Remove step length per second outliers during the gait sequence --> Hampel filter based on median absolute deviation
        duration = int(np.floor(len(vacc) / sampling_rate_hz))  # bottom-rounded duration (s)
        ICtime = initial_contacts / sampling_rate_hz  # initial contacts: samples -> sec
        sec_centers = (
            np.arange(0, duration) + 1
        )  # TODO: Differently from how it is done for cadence, here we consider second bins going from 0.5 to 1.5, from 1.5 to 2.5, etc...
        # padding last value of the calculated step length values so that sl_zjilstra_v3 is as long as ICtime
        if len(sl_zjilstra_v3) < len(ICtime):
            sl_zjilstra_v3 = np.concatenate(
                [sl_zjilstra_v3, np.tile(sl_zjilstra_v3[-1], len(ICtime) - len(sl_zjilstra_v3))]
            )

        slSec_zjilstra_v3 = robust_step_para_to_sec(
            ICtime, sl_zjilstra_v3, sec_centers, self.max_interpolation_gap_s, self.step_length_smoothing.clone()
        )
        # 11. Approximated stride length per second values = 2* step length per second values
        slSec = slSec_zjilstra_v3[0:duration] * 2

        # Results
        # Primary output: interpolated stride length per second
        self.stride_length_per_sec_list_ = pd.DataFrame(
            {"stride_length_m": slSec}, index=as_samples(sec_centers, sampling_rate_hz)
        ).rename_axis(index="sec_center_samples")
        # Secondary outputs
        self.step_length_list_ = sl_zjilstra_v3  # raw step length values
        self.step_length_per_sec_list_ = slSec_zjilstra_v3  # interpolated step length per second

        return self


def _zjilsV3(
    vacc_high: np.ndarray,
    sampling_rate_hz: float,
    step_length_scaling_factor: float,
    initial_contacts: np.ndarray,
    sensor_height_m: float,
) -> np.ndarray:
    """
    Step length estimation using the biomechanical model propose by Zijlstra & Hof [1]
    [1] Zijlstra, W., & Hof, A. L. (2003). Assessment of spatio-temporal gait parameters from trunk accelerations
    during human walking. Gait & posture, 18(2), 1-10.
    Inputs:
    - vacc_high: vertical acceleration recorded on lower back, high-pass filtered.
    - sampling_rate_hz: sampling frequency of the acc signal.
    - step_length_scaling_factor: tuning factor estimated by data from various clinical populations
    - initial_contacts: vector containing the timing of initial contacts events (samples)
    - sensor_height_m: Low Back height, i.e., the distance from ground to sensor location on lower back (in m)

    Output:
    - sl_zjilstra_v3: estimated step length.
    """
    # estimate vertical speed
    vspeed = -np.cumsum(vacc_high) / sampling_rate_hz
    # drift removal (high pass filtering)
    HP_filter = ButterworthFilter(order=4, cutoff_freq_hz=1, filter_type="highpass")
    speed_high = HP_filter.filter(vspeed, sampling_rate_hz=sampling_rate_hz).filtered_data_

    # speed_high = lfilter(b, a, vspeed)
    # estimate vertical displacement
    vdis_high_v2 = np.cumsum(speed_high) / sampling_rate_hz
    # initialize the output array as an array of zeros having one less element
    # than the array of ICs
    h_jilstra_v3 = np.zeros(len(initial_contacts) - 1)
    # initial estimates of the step length
    for k in range(len(initial_contacts) - 1):
        # the k-th step length value is initially estimated as the absolute difference between the maximum and the
        # minimum (range) of the vertical displacement between the k-th and (k+1)-th IC
        h_jilstra_v3[k] = np.abs(
            max(vdis_high_v2[initial_contacts[k] : initial_contacts[k + 1]])
            - min(vdis_high_v2[initial_contacts[k] : initial_contacts[k + 1]])
        )
    # biomechanical model formula
    sl_zjilstra_v3 = (
        step_length_scaling_factor * 2 * np.sqrt(np.abs((2 * sensor_height_m * h_jilstra_v3) - (h_jilstra_v3**2)))
    )
    return sl_zjilstra_v3
