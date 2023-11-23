from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.signal import cwt, ricker
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from tpcp import cf
from typing_extensions import Self, Unpack

from gaitlink.data_transform.base import BaseFilter
from gaitlink.data_transform._filter import EpflDedriftedGaitFilter
from gaitlink.data_transform._filter import EpflGaitFilter

# TODO: Some parameter renaming might be in order


@base_icd_docfiller
class SD_algo_AMC(BaseIcdDetector):
    """Implementation of the ICD algorithm by McCamley et al. (2012) [1]_ modified by Ionescu et al. (2020) [2]_

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


    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(gsd_list_)s

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

    pre_filter: Optional[BaseFilter] # IS IT RIGHT? --> Optional[EpflDedriftedGaitFilter]
    target_sampling_rate_hz: float
    def __init__(
        self,
        *,
        pre_filter: Optional[BaseFilter] = cf(EpflDedriftedGaitFilter()),
    ) -> None:
        self.pre_filter = pre_filter
        self.target_sampling_rate_hz = 40 # (Hz)

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

        # TODO: Add docstring
        relevant_columns = ["acc_x"] # acc_x: vertical acceleration
        accV = data[relevant_columns]
        accV = accV.to_numpy() # pd.Series--> ndarray
        def resampInterp(signal: np.ndarray, fs_initial: float, fs_final: float):
            recordingTime = len(signal)
            x = np.arange(1, recordingTime + 1)  # MATLAB uses 1-based indexing
            xq = np.arange(1, recordingTime + 1, fs_initial / fs_final)  # creates the new time vector
            interp_func = interp1d(x, signal,
                                   kind='linear',
                                   axis=0,
                                   fill_value='extrapolate')  # returns an interpolation function
            yResamp = interp_func(xq) # performs interpolation
            return yResamp

        accV40 = resampInterp(signal = accV, fs_initial = sampling_rate_hz, fs_final = self.target_sampling_rate_hz)
        # Get the coefficients of the gait filter (just to perform padding)
        # SHOULD I USE A FIXED LENGTH FOR PADDING (10000*102)
        b, a = EpflGaitFilter().coefficients
        # Padding for short data
        len_padd = 10000 * len(b)
        accV40_zp = np.pad(accV40, (len_padd, len_padd), 'wrap')
        # Band-pass filtering (SHOULD I USE transformed_data_ INSTEAD OF filtered_data_?)
        accV40_bpf = self.pre_filter.clone().filter(accV40_zp, sampling_rate_hz=self.target_sampling_rate_hz).filtered_data_
        # Remove the padding
        accV40_bpf_rmzp = accV40_bpf[len_padd - 1:-len_padd]
        # Cumulative integral
        accVLPInt = cumtrapz(accV40_bpf_rmzp, initial=0) / target_sampling_rate_hz
        # %% Continuous Wavelet Transform (CWT)
        accVLPIntCwt = cwt(accVLPInt, ricker, [6.4])
        # Remove the mean from accVLPIntCwt
        accVLPIntCwt = accVLPIntCwt - accVLPIntCwt.mean()
        # %% Detect ICs
        x = accVLPIntCwt.squeeze()  # squeeze extra dimensions
        # Detect the maximum peaks between the zero crossings
        def imax(x):
            # Return indices of maximum values
            return np.argmax(x)
        def MaxPeaksBetweenZC(x):
            # peaks and locations from vector x between zero crossings and location

            # Find zero crossing locations
            ix = np.asarray(np.abs(np.diff(np.sign(x))) == 2).nonzero()[0] + 1

            L = len(ix) - 1  # Number of zero crossings minus 1

            # Find the indices of maximum values between zero crossings
            ipk = [imax(np.abs(x[ix[i]:ix[i + 1]])) for i in range(L)]
            # the previous line basically considers L separate intervals between the L+1 zero crossings and returns the
            # index of the peak in each interval

            # Calculate the indices of the maximum values in the original vector
            ipks = [ix[i] + ipk[i] for i in range(L)]

            # Get the signed peaks
            pks = x[ipks]

            return pks, ipks

        pks, ipks = MaxPeaksBetweenZC(x)
        # Filter negative peaks
        indx = np.where(pks < 0)[0].tolist()
        ICs = np.array([ipks[i] for i in indx])
        self.ICs_ = ICs.copy()

        return self
