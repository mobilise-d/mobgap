import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from gaitmap.base import BaseOrientationMethod
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
    Moreover, differently from how it is done for cadence, here we consider second bins going from 0.5 to 1.5,
    from 1.5 to 2.5, etc...
    """,
        "sl_common_paras": """
    orientation_method
        The orientation method for aligning the sensor axes with the global reference system. If this argument is not
        passed, orientation is not performed (default value: None).
    step_length_smoothing
        The filter used to smooth the step length.
        This is used to remove outliers in the step length (e.g. when initial contacts are not detected).
        The filter is applied twice, once to the raw step length and a second time on the interpolated step length
        values per second.
        We recommend to use a Hampel filter for this.
    max_interpolation_gap_s
        The maximum gap in seconds that is interpolated.
        If the gap is larger than this value, the second is filled with NaNs.
        We don't fill "gaps" at the start and end of the recording, as we assume that gait sequences are cut anyway
        to start and end with a valid initial contact.
    step_length_scaling_factor
        The scaling factor for the Zijlstra biomechanical model.
        Possible attributes are step_length_scaling_factor_ms_ms and step_length_scaling_factor_ms_all.
    """,
    },
    doc_summary="Decorator to fill the explanation of the stride length per second interpolation.",
)


@sl_docfiller
class SlZijlstra(BaseSlCalculator):
    """
    Implementation of the stride length algorithm by Zijlstra and Hof (2003) [1]_
    modified by Soltani et al. (2021) [2]_.

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
    sensor_height_m
            Height of the sensor mounted on the lower-back in meters.

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

    .. [1] W. Zijlstra, & A. L. Hof, "Assessment of spatio-temporal gait parameters from trunk accelerations during
        human walking" Gait & posture, vol. 18, no. 2, pp. 1-10, 2003.
    .. [2] A. Soltani, et al. "Algorithms for walking speed estimation using a lower-back-worn inertial sensor:
        A cross-validation on speed ranges." IEEE TNSRE, vol 29, pp. 1955-1964, 2021.
    .. [3] https://github.com/mobilise-d/Mobilise-D-TVS-Recommended-Algorithms/blob/master/SLA_SLB/STRIDELEN.m

    """

    sensor_height_m: float

    orientation_method: Optional[BaseOrientationMethod]
    acc_smoothing: BaseFilter
    speed_smoothing: BaseFilter
    step_length_smoothing: BaseFilter
    max_interpolation_gap_s: float

    raw_step_length_: np.ndarray
    step_length_per_sec_: np.ndarray

    class PredefinedParameters:
        """
        Predefined factors for scaling the step length model.

        The step length scaling factor to be used in the biomechanical model.
        The reported parameters were previously accessed with model.K.ziljsV3.
        Coefficients are divided by four to separate the coefficient from
        the conversion from step length to stride length and from the x2 factor of the formula declared
        in [1, 2].
        """

        step_length_scaling_factor_ms_ms = {"step_length_scaling_factor": 4.587 / 4}
        step_length_scaling_factor_ms_all = {"step_length_scaling_factor": 4.739 / 4}

    # TODO: Decide how we want to handle the default values for the parameters
    @set_defaults(**PredefinedParameters.step_length_scaling_factor_ms_ms)
    def __init__(
        self,
        *,
        orientation_method: Optional[BaseOrientationMethod] = None,
        acc_smoothing: BaseFilter = cf(ButterworthFilter(order=4, cutoff_freq_hz=0.1, filter_type="highpass")),
        speed_smoothing: BaseFilter = cf(ButterworthFilter(order=4, cutoff_freq_hz=1, filter_type="highpass")),
        step_length_smoothing: BaseFilter = cf(HampelFilter(2, 3.0)),
        max_interpolation_gap_s: float = 3,
        step_length_scaling_factor: float,
    ) -> None:
        self.acc_smoothing = acc_smoothing
        self.speed_smoothing = speed_smoothing
        self.step_length_smoothing = step_length_smoothing
        self.max_interpolation_gap_s = max_interpolation_gap_s
        self.orientation_method = orientation_method
        self.step_length_scaling_factor = step_length_scaling_factor

    @sl_docfiller
    def calculate(
        self,
        data: pd.DataFrame,
        initial_contacts: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        sensor_height_m: float,
        **_: Unpack[dict[str, Any]],
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

        if not initial_contacts["ic"].is_monotonic_increasing:
            raise ValueError("Initial contacts must be sorted in ascending order.")
        # 1. Sensor alignment (optional): Madgwick complementary filter
        vacc = data[["acc_x"]]
        if self.orientation_method is not None:
            # perform rotation
            rotated_data = (
                self.orientation_method.clone().estimate(data, sampling_rate_hz=sampling_rate_hz).rotated_data_
            )
            vacc = rotated_data[["acc_x"]]  # consider acceleration
        # 2. High-pass filtering --> lower cut-off: 0.1 Hz, filter design: Butterworth IIR, order: 4
        vacc_filtered = self.acc_smoothing.clone().filter(vacc, sampling_rate_hz=sampling_rate_hz).filtered_data_
        # 3. Integration of vertical acceleration --> vertical speed
        # 4. Drift removal (high-pass filtering) --> lower cut-off: 1 Hz, filter design: Butterworth IIR, order: 4
        # 5. Integration of vertical speed --> vertical displacement d(t)
        # 6. Compute total vertical displacement during the step (d_step):
        # d_step = |max(d(t)) - min(d(t))|
        # 7. Biomechanical model:
        # StepLength = A * 2 * sqrt(2 * LBh * d_step - d_step^2)
        # A: step length scaling factor, optimized by training for each population
        # LBh: sensor height in meters, representative of the height of the center of mass
        duration = data.shape[0] / sampling_rate_hz
        sec_centers = np.arange(0, duration) + 0.5
        if len(initial_contacts) <= 1:
            # We can not calculate step length with only one initial contact
            warnings.warn("Can not calculate step length with only one or zero initial contacts.", stacklevel=3)
            stride_length_per_sec = np.full(len(sec_centers), np.nan)
            raw_step_length = np.full(len(initial_contacts), np.nan)
            step_length_per_sec = np.full(len(sec_centers), np.nan)

        else:
            initial_contacts = np.squeeze(initial_contacts["ic"].astype(int))
            raw_step_length = self._calc_step_length(vacc_filtered, initial_contacts)
            initial_contacts_per_sec = initial_contacts / sampling_rate_hz
            # 8. Remove step length outliers during the gait sequence --> Hampel filter based on median absolute
            #    deviation
            # 9. Interpolation of StepLength values --> Step length per second values
            # 10. Remove step length per second outliers during the gait sequence --> Hampel filter based on median
            # absolute deviation
            # We repeat the last step length to get the same number of step length as initial contacts for the
            # interpolation
            raw_step_length_padded = np.append(raw_step_length, raw_step_length[-1])
            step_length_per_sec = robust_step_para_to_sec(
                initial_contacts_per_sec,
                raw_step_length_padded,
                sec_centers,
                self.max_interpolation_gap_s,
                self.step_length_smoothing.clone(),
            )
            # 11. Approximated stride length per second values = 2 * step length per second values
            stride_length_per_sec = step_length_per_sec * 2
            # Results
        # Primary output: interpolated stride length per second
        self.stride_length_per_sec_ = pd.DataFrame(
            {"stride_length_m": stride_length_per_sec}, index=as_samples(sec_centers, sampling_rate_hz)
        ).rename_axis(index="sec_center_samples")
        # Secondary outputs
        self.raw_step_length_ = raw_step_length
        self.step_length_per_sec_ = step_length_per_sec
        return self

    def _calc_step_length(
        self,
        vacc_filtered: np.ndarray,
        initial_contacts: np.ndarray,
    ) -> np.ndarray:
        """Step length estimation using the biomechanical model propose by Zijlstra & Hof [1].

        Notes
        -----
        .. [1] Zijlstra, W., & Hof, A. L. (2003). Assessment of spatio-temporal gait parameters from trunk accelerations
        during human walking. Gait & posture, 18(2), 1-10.
        """
        vertical_speed = -np.cumsum(vacc_filtered) / self.sampling_rate_hz
        speed_filtered = (
            self.speed_smoothing.clone().filter(vertical_speed, sampling_rate_hz=self.sampling_rate_hz).filtered_data_
        )
        # estimate vertical displacement
        vertical_displacement = np.cumsum(speed_filtered) / self.sampling_rate_hz

        height_change_per_step = np.zeros(len(initial_contacts) - 1)

        for step_id, (step_start, step_end) in enumerate(zip(initial_contacts[:-1], initial_contacts[1:])):
            # np.ptp -> max - min
            height_change_per_step[step_id] = np.abs(np.ptp(vertical_displacement[step_start:step_end]))

        # biomechanical model formula (inverted Pendulum model)
        step_length = (
            self.step_length_scaling_factor
            * 2
            * np.sqrt(np.abs((2 * self.sensor_height_m * height_change_per_step) - (height_change_per_step**2)))
        )
        return step_length
