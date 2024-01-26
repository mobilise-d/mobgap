from typing import Any, Optional

import numpy as np
import pandas as pd
from pywt import cwt
from scipy.integrate import cumtrapz
from tpcp import cf
from typing_extensions import Self, Unpack

from gaitlink.consts import GRAV_MS2
from gaitlink.data_transform import EpflDedriftedGaitFilter, EpflGaitFilter, Resample
from gaitlink.data_transform.base import BaseFilter, BaseTransformer
from gaitlink.icd.base import BaseIcdDetector, base_icd_docfiller


@base_icd_docfiller
class IcdIonescu(BaseIcdDetector):
    """Implementation of the ICD algorithm by McCamley et al. (2012) [1]_ modified by Ionescu et al. (2020) [2]_.

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

    - The default value of the width of the cwt scale is lowered from 9 to 6.4. This method produces similar but
      different results. However, based on some testing, this parameter did not seem to have a big impact on the
      results, anyway.

    .. [1] J. McCamley, M. Donati, E. Grimpampi, C. Mazzà, "An enhanced estimate of initial contact and final contact
    instants of time using lower trunk inertial sensor data", Gait & Posture, vol. 36, no. 2, pp. 316-318, 2012.
    .. [2] A. Paraschiv-Ionescu, A. Soltani and K. Aminian, "Real-world speed estimation using single trunk IMU:
    methodological challenges for impaired gait patterns," 2020 42nd Annual International Conference of the IEEE
    Engineering in Medicine & Biology Society (EMBC), Montreal, QC, Canada, 2020, pp. 4596-4599,
    doi: 10.1109/EMBC44109.2020.9176281.
    .. [3] https://github.com/mobilise-d/Mobilise-D-TVS-Recommended-Algorithms/blob/master/ICDA/Library/SD_algo_AMC.m
    """

    pre_filter: Optional[BaseFilter]
    resampler: Optional[BaseTransformer]

    _INTERNAL_FILTER_SAMPLING_RATE_HZ: int = 40

    def __init__(
        self, *, pre_filter: Optional[BaseFilter] = cf(EpflDedriftedGaitFilter()), cwt_width: float = 9.0
    ) -> None:
        self.pre_filter = pre_filter
        self.cwt_width = cwt_width

    @base_icd_docfiller
    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_: Unpack[dict[str, Any]]) -> Self:
        """%(detect_short)s.

        Parameters
        ----------
        %(detect_para)s

        %(detect_return)s

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # 0. SELECT RELEVANT COLUMNS
        # For the ICD algorithm only vertical acceleration (i.e. x-component) is required.
        relevant_columns = ["acc_x"]  # acc_x: vertical acceleration
        acc_v = data[relevant_columns]
        acc_v = acc_v.to_numpy()  # pd.Series--> ndarray
        acc_v = acc_v / GRAV_MS2  # GRAV_MS2 = 9.81: [m/s^2]--> [g]

        # 1. RESAMPLING
        signal_downsampled = (
            Resample(self._INTERNAL_FILTER_SAMPLING_RATE_HZ)
            .transform(data=acc_v, sampling_rate_hz=sampling_rate_hz)
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
        acc_v_40_bpf = (
            self.pre_filter.clone()
            .filter(signal_downsampled_padded, sampling_rate_hz=self._INTERNAL_FILTER_SAMPLING_RATE_HZ)
            .filtered_data_.squeeze()
        )
        # Remove the padding
        acc_v_40_bpf_rmzp = acc_v_40_bpf[len_pad - 1 : -len_pad]
        # 3. CUMULATIVE INTEGRAL
        acc_v_lp_int = cumtrapz(acc_v_40_bpf_rmzp, initial=0) / self._INTERNAL_FILTER_SAMPLING_RATE_HZ
        # 4. CONTINUOUS WAVELET TRANSFORM (CWT)
        acc_v_lp_int_cwt, _ = cwt(
            acc_v_lp_int.squeeze(),
            [self.cwt_width],
            "gaus2",
            sampling_period=1 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ,
        )
        # Remove the mean from accVLPIntCwt
        acc_v_lp_int_cwt = acc_v_lp_int_cwt - acc_v_lp_int_cwt.mean()
        # 5. INTRA-ZERO-CROSSINGS PEAK DETECTION
        x = acc_v_lp_int_cwt.squeeze()
        # Detect the maximum peaks between the zero crossings
        pks, ipks = max_peaks_between_zc(x)
        # Filter negative peaks
        icd_array = np.array(ipks)[pks < 0]
        # Consider indices of the first and last elements of the input dataframe as ICs
        # icd_array = np.concatenate([np.array([1]),icd_array,np.array([data.shape[0]])])
        # TODO: Do we need the +1 here?
        detected_ics = icd_array + 1
        # OUTPUT: first and last element of the gsd are considered ICs by default
        detected_ics = pd.DataFrame({"ic": detected_ics}).rename_axis(index="ic_id")

        detected_ics_upsampeled = (
            (detected_ics * sampling_rate_hz / self._INTERNAL_FILTER_SAMPLING_RATE_HZ).round().astype(int)
        )
        self.ic_list_ = detected_ics_upsampeled

        return self


def max_peaks_between_zc(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Peaks and their locations between zero-crossings of array x
    #
    # Find zero crossing locations
    # TODO: This might not handle cases well, where the value is exactly zero
    ix = np.asarray(np.abs(np.diff(np.sign(x))) == 2).nonzero()[0] + 1
    n_zcs = len(ix) - 1  # Number of zero-crossings minus 1
    # Find the indices of maximum values between zero crossings. Code considers L separate intervals between the L+1
    # zero crossings and returns the index of the peak in each interval
    ipks = np.array([np.argmax(np.abs(x[ix[i] : ix[i + 1]])) + ix[i] for i in range(n_zcs)])
    # Get the signed peaks
    pks = x[ipks]
    return pks, ipks
