from typing import Any, Literal

import numpy as np
import pandas as pd
from gaitmap.data_transform import Resample
from numpy.linalg import norm
from pywt import cwt
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from typing_extensions import Self, Unpack

from gaitlink.data_transform import EpflDedriftedGaitFilter, EpflGaitFilter
from gaitlink.icd.base import BaseIcdDetector, base_icd_docfiller


@base_icd_docfiller
class IcdShinImproved(BaseIcdDetector):
    """Detect initial contacts using the Shin [1]_ algorithm, with improvements by Ionescu et al. [2]_.

    This algorithm is designed to detect initial contacts from accelerometer signals within a gait sequence.
    The algorithm filters the accelerometer signal down to its primary frequency components and then detects zero
    crossings in the filtered signal.
    These are interpreted as initial contacts.

    This is based on the implementation published as part of the mobilised project [3]_.
    However, this implementation deviates from the original implementation in some places.
    For details, see the notes section and the examples.

    Parameters
    ----------
    axis
        selecting which part of the accelerometer signal to be used. Can be 'x', 'y', 'z', or 'norm'.
        The default is 'norm', which is also the default in the original implementation.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(ic_list_)s
    final_filtered_signal_
        The downsampled signal after all filter steps.
        This might be useful for debugging.
    ic_list_internal_
        The initial contacts detected on the downsampled signal, before upsampling to the original sampling rate.
        This can be useful for debugging in combination with the `final_filtered_signal_` attribute.


    Notes
    -----
    Points of deviation from the original implementation and their reasons:

    - Configurable accelerometer signal: on matlab, all axes are used to calculate ICs, here we provide
      the option to select which axis to use. Despite the fact that the Shin algorithm on matlab uses all axes,
      here we provide the option of selecting a single axis because other contact detection algorithms use only the
      horizontal axis.
    - We use a different downsampling method, which should be "more" correct from a signal theory perspective,
      but will yield slightly different results.
    - The matlab code upsamples to 50 Hz before the final detection of 0-crossings.
      We skip the upsampling of the filtered signal and perform the 0-crossing detection on the downsampled signal.
      To compensate for the "loss of accuracy" due to the downsampling, we use linear interpolation to determine the
      exact position of the 0-crossing, even when it occurs between two samples.
      We then project the interpolated index back to the original sampling rate.
    - For CWT and gaussian filter, the actual parameter we pass to the respective functions differ from the matlab
      implementation, as the two languages use different implementations of the functions.
      However, the similarity of the output was visually confirmed.
    - All parameters are expressed in the units used in gaitlink, as opposed to matlab.
      Specifically, we use m/s^2 instead of g.
    - Some early testing indicates, that the python version finds all ICs 5-10 samples earlier than the matlab version.
      However, this seems to be a relatively consistent offset.
      Hence, it might be possible to shift/tune this in the future.

    .. [1] Shin, Seung Hyuck, and Chan Gook Park. "Adaptive step length estimation algorithm
       using optimal parameters and movement status awareness." Medical engineering & physics 33.9 (2011): 1064-1071.
    .. [2] Paraschiv-Ionescu, A. et al. "Real-world speed estimation using single trunk IMU:
       methodological challenges for impaired gait patterns". IEEE EMBC (2020): 4596-4599
    .. [3] https://github.com/mobilise-d/Mobilise-D-TVS-Recommended-Algorithms/blob/master/CADB_CADC/Library/Shin_algo_improved.m

    """

    axis: Literal["x", "y", "z", "norm"]

    _INTERNAL_FILTER_SAMPLING_RATE_HZ: int = 40

    final_filtered_signal_: np.ndarray
    ic_list_internal_: pd.DataFrame

    def __init__(self, axis: Literal["x", "y", "z", "norm"] = "norm") -> None:
        self.axis = axis

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

        if self.axis not in ["x", "y", "z", "norm"]:
            raise ValueError("Invalid axis. Choose 'x', 'y', 'z', or 'norm'.")

        signal = (
            norm(data[["acc_x", "acc_y", "acc_z"]].to_numpy(), axis=1)
            if self.axis == "norm"
            else data[f"acc_{self.axis}"].to_numpy()
        )

        # Resample to 40Hz to process with filters
        signal_downsampled = (
            Resample(self._INTERNAL_FILTER_SAMPLING_RATE_HZ)
            .transform(data=signal, sampling_rate_hz=sampling_rate_hz)
            .transformed_data_
        )

        # We need to intitialize the filter once to get the number of coefficients to calculate the padding.
        # This is not ideal, but works for now.
        # TODO: We should evaluate, if we need the padding at all, or if the filter methods that we use handle that
        #  correctly anyway. -> filtfilt uses padding automatically and savgol allows to actiavte padding, put uses the
        #  default mode (polyinomal interpolation) might be suffiecent anyway, cwt might have some edeeffects, but
        #  usually nothing to worry about.
        n_coefficients = len(EpflGaitFilter().coefficients[0])

        # Padding to cope with short data
        len_pad = 4 * n_coefficients
        signal_downsampled_padded = np.pad(signal_downsampled, (len_pad, len_pad), "wrap")

        # Filters
        # 1
        # TODO (future): Replace svagol and cwt with class implementation to easily expose parameters.
        tmp_sig_1 = savgol_filter(signal_downsampled_padded.squeeze(), window_length=21, polyorder=7)
        # 2
        tmp_sig_2 = (
            EpflDedriftedGaitFilter()
            .filter(tmp_sig_1, sampling_rate_hz=self._INTERNAL_FILTER_SAMPLING_RATE_HZ)
            .filtered_data_
        )
        # 3
        # NOTE: Original MATLAB code calls old version of cwt (open wavelet.internal.cwt in MATLAB to inspect) in
        #   accN_filt3=cwt(accN_filt2,10,'gaus2',1/40);
        #   Here, 10 is the scale, gaus2 is the second derivative of a Gaussian wavelet, aka a Mexican Hat or Ricker
        #   wavelet.
        #   In Python, a scale of 7 matches the MATLAB scale of 10 from visual inspection of plots (likely due to how to
        #   two languages initialise their wavelets), giving the line below
        ricker_width = 10
        tmp_sig_3, _ = cwt(
            tmp_sig_2.squeeze(), [ricker_width], "gaus2", sampling_period=1 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ
        )
        # 4
        tmp_sig_4 = savgol_filter(tmp_sig_3.squeeze(), window_length=11, polyorder=5)
        # 5
        tmp_sig_5, _ = cwt(
            tmp_sig_4.squeeze(), [ricker_width], "gaus2", sampling_period=1 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ
        )
        # Compared to matlab the python gauss filter needs the matlab window with divided by 5
        tmp_sig_6 = tmp_sig_5
        for sigma in [2, 2, 3]:
            tmp_sig_6 = gaussian_filter(tmp_sig_6.squeeze(), sigma)
        # Remove padding
        final_filtered = tmp_sig_6[len_pad:-len_pad]

        self.final_filtered_signal_ = final_filtered

        detected_ics = find_zero_crossings(final_filtered, "negative_to_positive")
        detected_ics = pd.DataFrame({"ic": detected_ics}).rename_axis(index="ic_id")

        self.ic_list_internal_ = detected_ics

        # Upsample initial contacts to original sampling rate
        detected_ics_upsampeled = (
            (detected_ics * sampling_rate_hz / self._INTERNAL_FILTER_SAMPLING_RATE_HZ).round().astype(int)
        )

        self.ic_list_ = detected_ics_upsampeled

        return self


def find_zero_crossings(
    signal: np.ndarray, mode: Literal["positive_to_negative", "negative_to_positive", "both"] = "both"
) -> np.ndarray:
    """
    Find zero crossings in a signal.

    Note, that the return values are floating point indices, as we use linear interpolation to refine the position of
    the zero crossing.

    Parameters
    ----------
    :param signal : np.ndarray
        A numpy array of the signal values.
    :param mode : str, optional
        A string specifying the type of zero crossings to detect.
        Can be 'positive_to_negative', 'negative_to_positive', or 'both'.
        The default is 'both'.

    Returns
    -------
    np.ndarray
        A numpy array containing the indices where zero crossings occur.

    Raises
    ------
    ValueError
        If the mode is not one of the specified options.

    Examples
    --------
    >>> signal = np.array([1, -1, 1, -1, 1])
    >>> find_zero_crossings(signal, mode="both")
    array([0, 1, 2, 3])

    Notes
    -----
    Under the hood we use :func:`numpy.sign` to find the indices where the sign of the signal changes.
    This function returns `-1 if x < 0, 0 if x==0, 1 if x > 0`.
    We change the output of the function to be `1 if x >= 0, -1 if x < 0`.
    This should handle cases where the value of the data becomes 0 for a couple of cases and then changes sign again.
    Note, that this might result in some unexpected behaviour, when the signal comes from negative values and then
    becomes 0 for a couple of samples and then negative again.
    This will be treated as two 0 crossings.
    However, if the same happens with positive values, no 0 crossing will be detected.
    With "real" data, this should not be a problem, but it is important to keep in mind.

    """
    sign = np.sign(signal)
    # We change the cases where the signal is 0 to be considered as positive.
    sign[sign == 0] = 1

    # Find indices where sign changes (this is the index before the 0 crossing)
    crossings = np.where(np.abs(np.diff(sign)))[0]

    if mode == "positive_to_negative":
        crossings = crossings[signal[crossings] > 0]
    elif mode == "negative_to_positive":
        crossings = crossings[signal[crossings] < 0]
    elif mode == "both":
        pass
    else:
        raise ValueError("Invalid mode. Choose 'positive_to_negative', 'negative_to_positive', or 'both'.")

    # Refine the position of the 0 crossing by linear interpolation and identify the real floating point index
    # of the 0 crossing.
    # Basically, we assume the real 0 crossing to be the index and the index + 1.
    # We compare the "absolute" value of the value at index and index + 1, to figure out how close to index the real
    # zero crossing is.
    refined_crossings = crossings + np.abs(signal[crossings] / (signal[crossings] - signal[crossings + 1]))

    return refined_crossings