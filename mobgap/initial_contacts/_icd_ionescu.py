from typing import Any

import numpy as np
import pandas as pd
from pywt import cwt
from scipy.integrate import cumulative_trapezoid
from tpcp import cf
from typing_extensions import Self, Unpack

from mobgap.data_transform import EpflDedriftedGaitFilter, EpflGaitFilter, Resample
from mobgap.data_transform.base import BaseFilter
from mobgap.initial_contacts._utils import find_zero_crossings
from mobgap.initial_contacts.base import BaseIcDetector, base_icd_docfiller


@base_icd_docfiller
class IcdIonescu(BaseIcDetector):
    """Implementation of the IC detection algorithm by McCamley (2012) [1]_ modified by Ionescu (2020) [2]_.

    The algorithm includes the following steps starting from vertical acceleration
    of the lower-back during a gait sequence:

    1. Resampling: 100 Hz --> 40 Hz
    2. Band-pass filtering --> lower cut-off: 0.15 Hz; higher cut-off: 3.14 Hz
    3. Cumulative integral --> cumulative trapezoidal integration
    4. Continuous Wavelet Transform (CWT) --> Ricker wavelet
    5. Zero crossings detection
    6. Detect peaks between zero crossings --> negative peaks = ICs

    This is based on the implementation published as part of the mobilised project [3]_.
    However, this implementation deviates from the original implementation in some places.
    For details, see the notes section and the examples.

    Parameters
    ----------
    pre_filter
        A pre-processing filter to apply to the data before the ICD algorithm is applied.
    cwt_width
        The width of the wavelet

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(ic_list_)s

    Notes
    -----
    Points of deviation from the original implementation and their reasons:

    - We use a different downsampling method, which should be "more" correct from a signal theory perspective,
      but will yield slightly different results.
    - We use a slightly different approach when it comes to the detection of the peaks between the zero crossings.
      However, the results of this step are identical to the matlab implementation.

    .. [1] J. McCamley, M. Donati, E. Grimpampi, C. Mazzà, "An enhanced estimate of initial contact and final contact
       instants of time using lower trunk inertial sensor data", Gait & Posture, vol. 36, no. 2, pp. 316-318, 2012.
    .. [2] A. Paraschiv-Ionescu, A. Soltani and K. Aminian, "Real-world speed estimation using single trunk IMU:
       methodological challenges for impaired gait patterns," 2020 42nd Annual International Conference of the IEEE
       Engineering in Medicine & Biology Society (EMBC), Montreal, QC, Canada, 2020, pp. 4596-4599,
       doi: 10.1109/EMBC44109.2020.9176281.
    .. [3] https://github.com/mobilise-d/Mobilise-D-TVS-Recommended-Algorithms/blob/master/ICDA/Library/SD_algo_AMC.m

    """

    pre_filter: BaseFilter
    cwt_width: float

    _INTERNAL_FILTER_SAMPLING_RATE_HZ: int = 40

    def __init__(self, *, pre_filter: BaseFilter = cf(EpflDedriftedGaitFilter()), cwt_width: float = 9.0) -> None:
        self.pre_filter = pre_filter
        self.cwt_width = cwt_width

    @base_icd_docfiller
    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_: Unpack[dict[str, Any]]) -> Self:
        """%(detect_short)s.

        %(detect_info)s

        Parameters
        ----------
        %(detect_para)s

        %(detect_return)s

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # 0. SELECT RELEVANT COLUMNS
        # For the ICD algorithm only vertical acceleration is required.
        acc_is = data[["acc_is"]].to_numpy()

        # 1. RESAMPLING
        signal_downsampled = (
            Resample(self._INTERNAL_FILTER_SAMPLING_RATE_HZ)
            .transform(data=acc_is, sampling_rate_hz=sampling_rate_hz)
            .transformed_data_.squeeze()
        )
        # 2. BAND-PASS FILTERING
        # Padding for short data: this is required because the length of the input vector must be greater
        # than padlen argument of the filtfilt function (3*max(b,a) = 306 by deafult), otherwise a
        # ValueError is raised.
        n_coefficients = len(EpflGaitFilter().coefficients[0])
        len_pad = 4 * n_coefficients
        signal_downsampled_padded = np.pad(signal_downsampled, (len_pad, len_pad), "wrap")
        # The clone() method is used in order to ensure that the new second instance is independent
        acc_is_40_bpf = (
            self.pre_filter.clone()
            .filter(signal_downsampled_padded, sampling_rate_hz=self._INTERNAL_FILTER_SAMPLING_RATE_HZ)
            .filtered_data_.squeeze()
        )
        # Remove the padding
        acc_is_40_bpf_rmzp = acc_is_40_bpf[len_pad - 1 : -len_pad]
        # 3. CUMULATIVE INTEGRAL
        acc_is_lp_int = cumulative_trapezoid(acc_is_40_bpf_rmzp, initial=0) / self._INTERNAL_FILTER_SAMPLING_RATE_HZ
        # 4. CONTINUOUS WAVELET TRANSFORM (CWT)
        acc_is_lp_int_cwt, _ = cwt(
            acc_is_lp_int.squeeze(),
            [self.cwt_width],
            "gaus2",
            sampling_period=1 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ,
        )
        acc_is_lp_int_cwt = acc_is_lp_int_cwt.squeeze()
        # Remove the mean from accVLPIntCwt
        acc_is_lp_int_cwt -= acc_is_lp_int_cwt.mean()

        # 5. INTRA-ZERO-CROSSINGS PEAK DETECTION
        # Detect the extrema between the zero crossings
        icd_array = _find_minima_between_zero_crossings(acc_is_lp_int_cwt)

        detected_ics = pd.DataFrame({"ic": icd_array}).rename_axis(index="step_id")
        detected_ics_unsampled = (
            (detected_ics * sampling_rate_hz / self._INTERNAL_FILTER_SAMPLING_RATE_HZ).round().astype("int64")
        )
        self.ic_list_ = detected_ics_unsampled

        return self


def _find_minima_between_zero_crossings(signal: np.ndarray) -> np.ndarray:
    zero_crossings = find_zero_crossings(signal, "both", refine=False).astype("int64")

    if len(zero_crossings) == 0:
        return np.array([])

    # We are then looking for minimas between the zero crossings.
    # Minimas can only happen between a positive to negative zero crossing and a negative to positive zero crossing.
    bool_map = signal[zero_crossings] >= 0
    # As the func returns the value before the 0 crossing, we need to add 1 to the indices to be able to correctly
    # select the range between the zero crossings.
    pos_to_neg = zero_crossings[bool_map] + 1
    neg_to_pos = zero_crossings[~bool_map] + 1
    # If the first zero crossing is a negative to positive zero crossing, we need to remove it.
    if not bool_map[0]:
        neg_to_pos = neg_to_pos[1:]

    minima = np.array([np.argmin(signal[start:end]) + start for start, end in zip(pos_to_neg, neg_to_pos)]).astype(
        "int64"
    )

    return minima
