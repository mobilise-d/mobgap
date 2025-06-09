import warnings
from types import MappingProxyType
from typing import Any, Final, Optional

import numpy as np
import pandas as pd
from tpcp import cf
from tpcp.misc import set_defaults
from typing_extensions import Self, Unpack

from mobgap.data_transform import ButterworthFilter, HampelFilter
from mobgap.data_transform.base import BaseFilter
from mobgap.orientation_estimation.base import BaseOrientationEstimation
from mobgap.stride_length.base import BaseSlCalculator, base_sl_docfiller
from mobgap.utils.conversions import as_samples
from mobgap.utils.dtypes import get_frame_definition
from mobgap.utils.interpolation import robust_step_para_to_sec


@base_sl_docfiller
class SlZijlstra(BaseSlCalculator):
    r"""Implementation of the stride length algorithm by Zijlstra (2003) [1]_ modified by Soltani (2021) [2]_.

    This algorithms uses an inverted pendulum model to estimate the step length.
    The step length is then "smoothed" using a robust outlier removal approach to deal with missing initial contacts.
    The output stride length is reported as the average for each 1 second bin within the data.

    For more details see the Notes section.

    This is based on the implementation published as part of the mobilised project [3]_.
    However, this implementation deviates from the original implementation in some places.
    For details, see the notes section and the examples.

    Parameters
    ----------
    step_length_scaling_factor
        The scaling factor for the Zijlstra biomechanical model.
        The final step length per step, will be scaled by this factor.
        Possible attributes are step_length_scaling_factor_ms_ms and step_length_scaling_factor_ms_all.
    acc_smoothing
        Lowpass filter applied to the vertical acceleration before integration.
    speed_smoothing
        Lowpass filter applied to the calculated vertical velocity for integration.
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
    orientation_method
        The orientation method for aligning the sensor axes with the global reference system. If this argument is not
        passed, orientation is not performed (default value: None).

    Other Parameters
    ----------------
    %(other_parameters)s
    sensor_height_m
            Height of the sensor mounted on the lower-back in meters.

    Attributes
    ----------
    %(stride_length_per_sec_)s
    raw_step_length_per_step_
        Secondary output.
        It provides a copy of the passed initial contact df with an additional column ``step_length_m`` that contains
        the estimated step length for each step.
        Note, that the df is one row shorter thant the passed list of ICs, as there is one less actual step then ICs.
    step_length_per_sec_
        Secondary output.
        A pandas dataframe of the same shape as ``stride_length_per_sec_``, but containing the intermediate step length
        values in a column called ``step_length_m``

    Notes
    -----
    The core algorithm performs the following steps:

    1. Sensor alignment (optional): Madgwick complementary filter
    2. High-pass filtering --> lower cut-off: 0.1 Hz, filter design: Butterworth IIR, filter order: 4
    3. Integration of vertical acceleration --> vertical speed
    4. Drift removal (high-pass filtering) --> lower cut-off: 1 Hz, filter design: Butterworth IIR, filter order: 4
    5. Integration of vertical speed --> vertical displacement d(t)
    6. Compute total vertical displacement during the step (d_step):

       .. math::
          d_step = |max(d(t)) - min(d(t))|

    7. Biomechanical model:

       .. math::
          \text{StepLength} = A * 2 * \sqrt{2 * LBh * d_{step} - d_{step}^2}

       A
        tuning coefficient, optimized by grid search.
       LBh
        sensor height in meters, representative of the height of the center of mass.

    This output is then further post-processed and interpolated to second bins:

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
    5. We multiply by 2 to convert from step length per second to stride length per second.
       This is an approximation and will not reflect the exact stride length of any individual stride.

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

    .. [1] W. Zijlstra, & A. L. Hof, "Assessment of spatio-temporal gait parameters from trunk accelerations during
        human walking" Gait & posture, vol. 18, no. 2, pp. 1-10, 2003.
    .. [2] A. Soltani, et al. "Algorithms for walking speed estimation using a lower-back-worn inertial sensor:
        A cross-validation on speed ranges." IEEE TNSRE, vol 29, pp. 1955-1964, 2021.
    .. [3] https://github.com/mobilise-d/Mobilise-D-TVS-Recommended-Algorithms/blob/master/SLA_SLB/STRIDELEN.m

    """

    sensor_height_m: float

    orientation_method: Optional[BaseOrientationEstimation]
    acc_smoothing: BaseFilter
    speed_smoothing: BaseFilter
    step_length_smoothing: BaseFilter
    max_interpolation_gap_s: float

    raw_step_length_per_step_: pd.DataFrame
    step_length_per_sec_: pd.DataFrame

    class PredefinedParameters:
        """
        Predefined factors for scaling the step length model.

        The step length scaling factor to be used in the biomechanical model.
        The provided values are optimized based on the MsProject dataset as part of the Mobilise-D project.

        Attributes
        ----------
        step_length_scaling_factor_ms_ms
            Optimized factor based on all MS patients within the MsProject dataset.
            Default value for the ``SlZijlstra`` algorithm.
        step_length_scaling_factor_ms_all
            Optimized factor based on ALL participants within the MsProject dataset.

        Examples
        --------
        >>> SlZijlstra(
        ...     max_interpolation_gap_s=2,
        ...     **SlZijlstra.PredefinedParameters.step_length_scaling_factor_ms_ms,
        ... )

        """

        step_length_scaling_factor_ms_ms: Final = MappingProxyType({"step_length_scaling_factor": 4.587 / 4})
        step_length_scaling_factor_ms_all: Final = MappingProxyType({"step_length_scaling_factor": 4.739 / 4})

    # TODO: Double check if we should have MS_MS or MS_ALL as default
    @set_defaults(**PredefinedParameters.step_length_scaling_factor_ms_ms)
    def __init__(
        self,
        *,
        orientation_method: Optional[BaseOrientationEstimation] = None,
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

    @base_sl_docfiller
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

        ic_list = initial_contacts["ic"].to_numpy()

        if len(ic_list) > 0 and (ic_list[0] != 0 or ic_list[-1] != len(data) - 1):
            warnings.warn(
                "Usually we assume that gait sequences are cut to the first and last detected initial "
                "contact. "
                "This is not the case for the passed initial contacts and might lead to unexpected "
                "results in the cadence calculation. "
                "Specifically, you will get NaN values at the start and the end of the output.",
                stacklevel=1,
            )

        frame = get_frame_definition(data, ["body", "global_body"])

        # 1. Sensor alignment (optional): Madgwick complementary filter
        if self.orientation_method is not None:
            if frame == "global_body":
                raise ValueError(
                    "The data already seems to be in the global frame based on the available columns. "
                    "Additional orientation estimation is not possible. "
                    "Set `orientation_estimation` to None."
                )

            # perform rotation
            rotated_data = (
                self.orientation_method.clone().estimate(data, sampling_rate_hz=sampling_rate_hz).rotated_data_
            )
            vacc = rotated_data[["acc_gis"]]  # consider acceleration
        else:
            vacc = data[["acc_is"]]

        duration = data.shape[0] / sampling_rate_hz
        sec_centers = np.arange(0, duration) + 0.5

        if len(ic_list) <= 1:
            # We can not calculate step length with only one initial contact
            warnings.warn("Can not calculate step length with only one or zero initial contacts.", stacklevel=1)
            self._set_all_nan(sec_centers, ic_list)
            return self

        # 2. High-pass filtering --> lower cut-off: 0.1 Hz, filter design: Butterworth IIR, order: 4
        try:
            vacc_filtered = self.acc_smoothing.clone().filter(vacc, sampling_rate_hz=sampling_rate_hz).filtered_data_
        except ValueError as e:
            if "padlen" in str(e):
                warnings.warn("Data is too short for the filter. Returning empty stride length results.", stacklevel=1)
                self._set_all_nan(sec_centers, ic_list)
                return self
            raise e from None

        # 3. Integration of vertical acceleration --> vertical speed
        # 4. Drift removal (high-pass filtering) --> lower cut-off: 1 Hz, filter design: Butterworth IIR, order: 4
        # 5. Integration of vertical speed --> vertical displacement d(t)
        # 6. Compute total vertical displacement during the step (d_step):
        # d_step = |max(d(t)) - min(d(t))|
        # 7. Biomechanical model:
        # StepLength = A * 2 * sqrt(2 * LBh * d_step - d_step^2)
        # A: step length scaling factor, optimized by training for each population
        raw_step_length = self._calc_step_length(vacc_filtered, ic_list)

        # We repeat the last step length to get the same number of step length as initial contacts for the
        # interpolation
        raw_step_length_padded = np.append(raw_step_length, raw_step_length[-1])
        # 8. Remove step length outliers during the gait sequence --> Hampel filter based on median absolute
        #    deviation
        # 9. Interpolation of StepLength values --> Step length per second values
        # 10. Remove step length per second outliers during the gait sequence --> Hampel filter based on median
        # absolute deviation
        initial_contacts_per_sec = ic_list / sampling_rate_hz
        step_length_per_sec = robust_step_para_to_sec(
            initial_contacts_per_sec,
            raw_step_length_padded,
            sec_centers,
            self.max_interpolation_gap_s,
            self.step_length_smoothing.clone(),
        )
        # 11. Approximated stride length per second values = 2 * step length per second values
        stride_length_per_sec = step_length_per_sec * 2

        self._unify_and_set_outputs(raw_step_length, step_length_per_sec, stride_length_per_sec, sec_centers)
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

    def _set_all_nan(self, sec_centers: np.ndarray, ic_list: np.ndarray) -> None:
        stride_length_per_sec = np.full(len(sec_centers), np.nan)
        raw_step_length = np.full(np.clip(len(ic_list) - 1, 0, None), np.nan)
        step_length_per_sec = np.full(len(sec_centers), np.nan)
        self._unify_and_set_outputs(raw_step_length, step_length_per_sec, stride_length_per_sec, sec_centers)

    def _unify_and_set_outputs(
        self,
        raw_step_length: np.ndarray,
        step_length_per_sec: np.ndarray,
        stride_length_per_sec: np.ndarray,
        sec_centers: np.ndarray,
    ) -> None:
        self.raw_step_length_per_step_ = self.initial_contacts.iloc[:-1].assign(step_length_m=raw_step_length)
        index = pd.Index(as_samples(sec_centers, self.sampling_rate_hz), name="sec_center_samples")
        self.step_length_per_sec_ = pd.DataFrame({"step_length_m": step_length_per_sec}, index=index)
        self.stride_length_per_sec_ = pd.DataFrame({"stride_length_m": stride_length_per_sec}, index=index)
