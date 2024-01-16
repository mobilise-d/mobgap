from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz
from scipy.signal import cwt, ricker
from tpcp import cf
from typing_extensions import Self, Unpack

from gaitlink.consts import GRAV_MS2
from gaitlink.data_transform._filter import EpflDedriftedGaitFilter
from gaitlink.data_transform._resample import Resample
from gaitlink.data_transform.base import BaseFilter, BaseTransformer
from gaitlink.icd.base import BaseIcdDetector, base_icd_docfiller

# TODO: Some parameter renaming might be in order


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

    # ALL PARAMETERS THAT ARE NOT DATA-RELATED IN THE INIT METHOD
    pre_filter: Optional[BaseFilter]  # IS IT RIGHT? --> Optional[EpflDedriftedGaitFilter]
    resampler: Optional[BaseTransformer]
    target_sampling_rate_hz: float

    def __init__(
        self, *, pre_filter: Optional[BaseFilter] = cf(EpflDedriftedGaitFilter()), cwt_width: float = 6.4
    ) -> None:
        self.pre_filter = pre_filter
        self.cwt_width = cwt_width

    @base_icd_docfiller
    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_: Unpack[dict[str, Any]]) -> Self:
        """%(detect_short)s. It is a summary of the detect method.

        Parameters
        ----------
        %(detect_para)s # input parameters

        %(detect_return)s # output parameters

        """
        self.data = data

        # 0. SELECT RELEVANT COLUMNS
        # For the ICD algorithm only vertical acceleration (i.e. x-component) is required.
        relevant_columns = ["acc_x"]  # acc_x: vertical acceleration
        acc_v = data[relevant_columns]
        acc_v = acc_v.to_numpy()  # pd.Series--> ndarray
        acc_v = acc_v / GRAV_MS2  # GRAV_MS2 = 9.81: [m/s^2]--> [g]

        # 1. RESAMPLING
        target_sampling_rate_hz = 40
        # Create an instance of the Resample class with the target sampling rate
        resampler = Resample(target_sampling_rate_hz)
        # Perform the resampling operation by calling the transform method
        resampled = resampler.transform(acc_v, sampling_rate_hz=sampling_rate_hz)
        acc_v_40 = resampled.transformed_data_
        # 2. BAND-PASS FILTERING
        # Padding for short data: this is required because the length of the input vector must be greater
        # than padlen argument of the filtfilt function (3*max(b,a) = 306 by deafult), otherwise a
        # ValueError is raised.
        len_padd = 4 * 102  # len(b) = 102.
        acc_v_40_zp = np.pad(acc_v_40.squeeze(), (len_padd, len_padd), "wrap")
        # The clone() method is used in order to ensure that the new second instance is independent
        acc_v_40_bpf = (
            self.pre_filter.clone()
            .filter(acc_v_40_zp, sampling_rate_hz=target_sampling_rate_hz)
            .filtered_data_.squeeze()
        )
        # Remove the padding
        acc_v_40_bpf_rmzp = acc_v_40_bpf[len_padd - 1 : -len_padd]
        # 3. CUMULATIVE INTEGRAL
        acc_v_lp_int = cumtrapz(acc_v_40_bpf_rmzp, initial=0) / target_sampling_rate_hz
        # 4. CONTINUOUS WAVELET TRANSFORM (CWT)
        acc_v_lp_int_cwt = cwt(acc_v_lp_int.squeeze(), ricker, [self.cwt_width])
        # Remove the mean from accVLPIntCwt
        acc_v_lp_int_cwt = acc_v_lp_int_cwt - acc_v_lp_int_cwt.mean()
        # 5. INTRA-ZERO-CROSSINGS PEAK DETECTION
        x = acc_v_lp_int_cwt.squeeze()  # squeeze extra dimensions
        # Detect the maximum peaks between the zero crossings
        pks, ipks = max_peaks_between_zc(x)
        # Filter negative peaks
        icd_array = np.array(ipks)[pks < 0]
        # Consider indices of the first and last elements of the input dataframe as ICs
        # icd_array = np.concatenate([np.array([1]),icd_array,np.array([data.shape[0]])])
        # Convert to seconds.
        icd_array = (icd_array + 1) / target_sampling_rate_hz
        # OUTPUT: first and last element of the gsd are considered ICs by default
        self.icd_list_ = pd.Series(data=icd_array, name="ic")
        return self


def max_peaks_between_zc(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Peaks and their locations between zero-crossings of array x
    #
    # Find zero crossing locations
    ix = np.asarray(np.abs(np.diff(np.sign(x))) == 2).nonzero()[0] + 1
    n_zcs = len(ix) - 1  # Number of zero-crossings minus 1
    # Find the indices of maximum values between zero crossings. Code considers L separate intervals between the L+1
    # zero crossings and returns the index of the peak in each interval
    ipks = [np.argmax(np.abs(x[ix[i] : ix[i + 1]])) + ix[i] for i in range(n_zcs)]
    # Get the signed peaks
    pks = x[ipks]
    return pks, ipks
