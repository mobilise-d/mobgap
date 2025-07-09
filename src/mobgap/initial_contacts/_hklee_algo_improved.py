from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.ndimage import grey_closing, grey_opening
from typing_extensions import Self, Unpack

from mobgap.consts import BF_ACC_COLS, SF_ACC_COLS
from mobgap.data_transform import (
    CwtFilter,
    EpflDedriftedGaitFilter,
    EpflGaitFilter,
    GaussianFilter,
    Pad,
    Resample,
    SavgolFilter,
    chain_transformers,
)
from mobgap.initial_contacts.base import BaseIcDetector, base_icd_docfiller
from mobgap.utils.dtypes import assert_is_sensor_data, get_frame_definition


@base_icd_docfiller
class IcdHKLeeImproved(BaseIcDetector):
    """Detect initial contacts using the HKLee [1]_ algorithm, with improvements by Ionescu et al. [2]_.

    This algorithm is designed to detect initial contacts from accelerometer signals within a gait sequence.
    The algorithm filters the accelerometer signal down to its primary frequency components and then employs
    morphological operations with closing and opening structural elements to detect signal closings and openings,
    respectively.
    Their difference is analyzed to identify instances where R is greater than 0.
    These regions are interpreted as initial contacts.

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
        (upsampled again in HKLee)
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
    - For CWT and gaussian filter, the actual parameter we pass to the respective functions differ from the matlab
      implementation, as the two languages use different implementations of the functions.
      However, the similarity of the output was visually confirmed.
    - The "closing" and "opening" operations are not exactly the same as in the original matlab code.
      This leads to small differences at the edges of the final signal.
      This can lead to one additional or one missing IC at the edges of the signal.
    - All parameters are expressed in the units used in mobgap, as opposed to matlab.
      Specifically, we use m/s^2 instead of g.

    Future work:

    - The algorithm can be improved by increasing the threshold of the allowed non-zero values.
      Currently, only single non-zero sequences are removed.
      For example, we could include a threshold of the minimum duration (samples) of an initial contact.


    .. [1] Lee, H-K., et al. "Computational methods to detect step events for normal and pathological
        gait evaluation using accelerometer." Electronics letters 46.17 (2010): 1185-1187.
    .. [2] Paraschiv-Ionescu, A. et al. "Real-world speed estimation using single trunk IMU:
       methodological challenges for impaired gait patterns". IEEE EMBC (2020): 4596-4599
    .. [3] https://github.com/mobilise-d/Mobilise-D-TVS-Recommended-Algorithms/blob/master/CADB_CADC/Library/HKLee_algo_improved.m

    """

    axis: Literal["is", "ml", "pa", "norm"]

    _INTERNAL_FILTER_SAMPLING_RATE_HZ: int = 40
    _UPSAMPLED_SAMPLING_RATE_HZ: int = 120

    final_filtered_signal_: np.ndarray
    ic_list_internal_: pd.DataFrame

    def __init__(self, axis: Literal["is", "ml", "pa", "norm"] = "norm") -> None:
        self.axis = axis

    @base_icd_docfiller
    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_: Unpack[dict[str, Any]]) -> Self:
        """%(detect_short)s.

        %(detect_info)s

        .. note:: As all other IC algorithms this algorithm technically only allows body-frame data as input.
            However, as this specific algorithm can run on the norm, we also allow sensor frame data if `self.axis ==
            "norm"`.

        Parameters
        ----------
        %(detect_para)s

        %(detect_return)s

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        if self.axis not in ["is", "ml", "pa", "norm"]:
            raise ValueError('Invalid axis. Choose ["is", "ml", "pa", "norm"].')

        if self.axis != "norm":
            assert_is_sensor_data(data, "body")
            signal = data[f"acc_{self.axis}"].to_numpy()
        else:
            # In case of norm, we support either body frame or sensor frame input.
            frame = get_frame_definition(data, ["sensor", "body"])
            axis = SF_ACC_COLS if frame == "sensor" else BF_ACC_COLS
            signal = norm(data[axis].to_numpy(), axis=1)

        # We need to initialize the filter once to get the number of coefficients to calculate the padding.
        # This is not ideal, but works for now.
        # TODO: We should evaluate, if we need the padding at all, or if the filter methods that we use handle that
        #  correctly anyway. -> filtfilt uses padding automatically and savgol allows to actiavte padding, put uses the
        #  default mode (polyinomal interpolation) might be suffiecent anyway, cwt might have some edeeffects, but
        #  usually nothing to worry about.
        n_coefficients = len(EpflGaitFilter().coefficients[0])

        # Padding to cope with short data
        len_pad_s = 4 * n_coefficients / self._INTERNAL_FILTER_SAMPLING_RATE_HZ
        padding = Pad(pad_len_s=len_pad_s, mode="wrap")

        if len(data) < 0.5 * sampling_rate_hz:
            # The threshold of 0.5 seconds is arbitrary, but it is a reasonable minimum length for gait data.
            # If the data is shorter than 0.5 seconds, we cannot apply the filter chain.
            # We just return the original signal.
            self.ic_list_internal_ = pd.DataFrame(columns=["ic"]).rename_axis(index="step_id")
            self.ic_list_ = pd.DataFrame(columns=["ic"]).rename_axis(index="step_id")
            return self

        # Now we build everything together into one filter chain.
        filter_chain = [
            # Resample to 40Hz to process with filters
            ("resampling", Resample(self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
            ("padding", padding),
            # Savgol filters
            # The original Matlab code useses two savgol filter in the chain.
            # To replicate them with our classes we need to convert the sample-parameters of the original matlab code to
            # sampling-rate independent units used for the parameters of our classes.
            # The parameters from the matlab code are: (21, 7) and (11, 5) for the 2 savgol filters
            (
                "savgol_1",
                SavgolFilter(
                    window_length_s=21 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ,
                    polyorder_rel=7 / 21,
                ),
            ),
            ("epfl_gait_filter", EpflDedriftedGaitFilter()),
            #   CWT - Filter
            #   Original MATLAB code calls old version of cwt (open wavelet.internal.cwt in MATLAB to inspect) in
            #   accN_filt3=cwt(accN_filt2,10,'gaus2',1/40);
            #   Here, 10 is the scale, gaus2 is the second derivative of a Gaussian wavelet, aka a Mexican Hat or Ricker
            #   wavelet.
            #   This frequency this scale corresponds to depends on the sampling rate of the data.
            #   As the mobgap cwt method uses the center frequency instead of the scale, we need to calculate the
            #   frequency that scale corresponds to at 40 Hz sampling rate.
            #   Turns out that this is 1.2 Hz
            ("cwt_1", CwtFilter(wavelet="gaus2", center_frequency_hz=1.2)),
            (
                "savol_2",
                SavgolFilter(
                    window_length_s=11 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ,
                    polyorder_rel=5 / 11,
                ),
            ),
            ("cwt_2", CwtFilter(wavelet="gaus2", center_frequency_hz=1.2)),
            ("gaussian_1", GaussianFilter(sigma_s=2 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
            ("gaussian_2", GaussianFilter(sigma_s=2 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
            ("gaussian_3", GaussianFilter(sigma_s=3 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
            ("padding_remove", padding.get_inverse_transformer()),
            ("resampling_up", Resample(self._UPSAMPLED_SAMPLING_RATE_HZ)),
        ]

        final_filtered = chain_transformers(signal, filter_chain, sampling_rate_hz=sampling_rate_hz)
        self.final_filtered_signal_ = final_filtered

        # Note that this is slightly different to how matlab is handling edges.
        # According to the closing source code in Matlab, the edges are padded by half the structure size with 0.
        # Opening does not seem to use any padding.
        # The closest we can get to this is using the "constant" mode in Python, which should also use 0 as padding.
        # However, the start and the end of the output still sometimes look the same resulting in either one additional
        # or one missing ICs at the edges.
        c = grey_closing(self.final_filtered_signal_, size=32, mode="constant")
        o = grey_opening(c, size=18, mode="constant")
        r = c - o

        final_detected_ics = pd.DataFrame(columns=["ic"]).rename_axis(index="step_id")

        if np.any(r > 0):
            non_zero = groupfind(r > 0)
            detected_ics = []
            for start, end in non_zero:
                values_within_range = r[start : end + 1]
                imax = start + np.argmax(values_within_range)
                detected_ics.append(imax)

            final_detected_ics["ic"] = detected_ics

        self.ic_list_internal_ = final_detected_ics

        # Downsample initial contacts to original sampling rate
        ic_downsampled = (
            (final_detected_ics * sampling_rate_hz / self._UPSAMPLED_SAMPLING_RATE_HZ).round().astype("int64")
        )

        self.ic_list_ = ic_downsampled

        return self


def groupfind(bool_array: np.ndarray) -> np.ndarray:
    """Find sequences in a bool array which are TRUE/1 and converts those to start-end indices.

    At least 2 consecutive True elements are required to be considered a sequence.
    The end index is the last element of the True-region

    Parameters
    ----------
    bool_array : array with shape (n,)
        boolean array with either 0/1, 0.0/1.0 or True/False elements

    Returns
    -------
    array of [start, end] indices with shape (n,2)

    Examples
    --------
    >>> example_array = np.array([0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0])
    >>> start_end_list = groupfind(example_array)
    >>> start_end_list
    array[[ 5  6]
        [ 9 11]]
    """
    if not isinstance(bool_array, np.ndarray):
        raise TypeError("Input must be a numpy array")

    if not np.array_equal(bool_array, bool_array.astype(bool)):
        raise ValueError("Input must be boolean array")

    if len(bool_array) == 0:
        return np.array([])

    nonzero = np.where(bool_array)[0]  # Find non-zeros
    endzero = np.where(np.diff(nonzero) > 1)[0]  # Find end of non-zero groups
    seq = np.zeros(
        (len(endzero) + 1, 2), dtype="int64"
    )  # Initializing array (+1 because last sequence is not calculated))
    seq[:, 1] = nonzero[np.append(endzero, -1)]  # End
    seq[:, 0] = nonzero[np.insert(endzero, 0, -1) + 1]  # Start
    seq = seq[seq[:, 1] - seq[:, 0] != 0]
    return seq
