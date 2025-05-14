from typing import Any, Literal

import numpy as np
import pandas as pd
from tpcp import cf
from typing_extensions import Self, Unpack

from mobgap.data_transform import ButterworthFilter
from mobgap.data_transform.base import BaseFilter
from mobgap.laterality.base import BaseLRClassifier, _unify_ic_lr_list_df, base_lrc_docfiller
from mobgap.utils.dtypes import assert_is_sensor_data


@base_lrc_docfiller
class LrcMcCamley(BaseLRClassifier):
    """McCamley algorithm for laterality detection of initial contacts.

    The McCamley algorithm [1]_ uses the sign of the angular velocity either yaw (rotation around IC-axis), the roll
    (rotation around PA-axis), or a combination of both as the distinguishing factor for identifying left and right ICs.

    For this the respective signal is filtered (high-pass to remove DC offset and low-pass to remove noise) and the sign
    at the position of the IC is used to determine the laterality.

    The original algorithm uses the yaw signal, but Ullrich et al. [2]_ showed that a combination of yaw and roll
    signals can improve the detection accuracy.
    Further, instead of simple mean subtraction, Ullrich et al. [2]_ used a Butterworth bandpass filter to smooth the
    signal.

    Parameters
    ----------
    axis
        The axis to use for the laterality detection. Can be one of "yaw", "roll", or "combined".
        For "roll" and "combined" the sign of the "roll" signal is inverted, as it is phase shifted compared to the yaw
        signal.
    smoothing_filter
        The filter to use for smoothing the signal. The filter should be a high-pass filter to remove the DC offset and
        a low-pass filter to remove noise.
        So any form of bandpass filter should be suitable.

    Attributes
    ----------
    %(ic_lr_list_)s
    smoothed_data_
        The smoothed data used for the laterality detection.
        This might be helpful for debugging or further analysis.

    Other Parameters
    ----------------
    %(other_parameters)s

    Notes
    -----
    In the edge case of data == 0, the left side is assumed.


    .. [1] J. McCamley et al., “An enhanced estimate of initial contact and final contact instants of time using lower
        trunk inertial sensor data,” Gait & posture, 2012, available at:
        https://www.sciencedirect.com/science/article/pii/S0966636212000707?via%%3Dihub

    .. [2] Ullrich M, Küderle A, Reggi L, Cereatti A, Eskofier BM, Kluge F. Machine learning-based distinction of left
        and right foot contacts in lower back inertial sensor gait data. Annu Int Conf IEEE Eng Med Biol Soc. 2021,
        available at: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9630653
    """

    axis: Literal["is", "pa", "combined"]
    smoothing_filter: BaseFilter

    smoothed_data_: pd.Series

    def __init__(
        self,
        axis: Literal["is", "pa", "combined"] = "combined",
        smoothing_filter: BaseFilter = cf(ButterworthFilter(order=4, cutoff_freq_hz=(0.5, 2), filter_type="bandpass")),
    ) -> None:
        self.axis = axis
        self.smoothing_filter = smoothing_filter

    @base_lrc_docfiller
    def predict(
        self,
        data: pd.DataFrame,
        ic_list: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """%(predict_short)s.

        Parameters
        ----------
        %(predict_para)s

        %(predict_return)s
        """
        self.sampling_rate_hz = sampling_rate_hz
        self.data = data
        self.ic_list = ic_list

        assert_is_sensor_data(data, frame="body")

        if data.empty or ic_list.empty:
            self.ic_lr_list_ = (
                pd.DataFrame(columns=["ic", "lr_label"], index=ic_list.index).dropna().pipe(_unify_ic_lr_list_df)
            )
            return self

        # create a copy of ic_list, otherwise they will get modified when adding the predicted labels
        # We also remove the "lr_label" column, if it exists, to avoid conflicts
        ic_list = ic_list.copy().drop(columns="lr_label", errors="ignore")

        if self.axis == "is":
            selected_data = data["gyr_is"]
        elif self.axis == "pa":
            # roll is phase shifted compared to yaw -> invert the sign
            selected_data = data["gyr_pa"] * -1
        elif self.axis == "combined":
            # combine both signals to amplify the differences between left and right steps
            selected_data = data["gyr_pa"] * -1 + data["gyr_is"]
        else:
            raise ValueError(f'Invalid axis configuration: {self.axis}. Allowed values are ["is", "pa", "combined"]')

        # The use of the smoothing filter is an addition made by Ullrich et al. to the original McCamley algorithm.
        # Originally, simply the mean of the signal was subtracted from the signal
        self.smoothed_data_ = (
            self.smoothing_filter.clone().filter(selected_data, sampling_rate_hz=self.sampling_rate_hz).filtered_data_
        )

        data_at_ic = self.smoothed_data_.iloc[ic_list["ic"].to_numpy()]

        ic_list["lr_label"] = np.where(data_at_ic <= 0, "left", "right")

        self.ic_lr_list_ = _unify_ic_lr_list_df(ic_list)
        return self
