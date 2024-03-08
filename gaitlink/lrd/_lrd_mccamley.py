from typing import Any, Literal

import numpy as np
import pandas as pd
from tpcp import cf
from typing_extensions import Self, Unpack

from gaitlink.data_transform import ButterworthFilter
from gaitlink.data_transform.base import BaseFilter
from gaitlink.lrd.base import BaseLRDetector, base_lrd_docfiller


class LrdMcCamley(BaseLRDetector):
    """
        This class uses the McCamley algorithm (later improved in Ullrich et al.) in order to predict whether each pre-determined initial contact (IC) corresponds to a left or a right step.

        In the original McCamley algorithm, the angular velocity around the vertical axis ("gyr_x") serves as the distinguishing factor for identifying left and right ICs. The process involves the following steps:

            * Signal Pre-processing: Subtracting the signal mean and applying a low-pass filter (4th order Butterworth filter with a 2 Hz cut-off frequency).
            * IC Assignment: Analyzing the sign of the filtered "gyr_x" value at the IC time point for classification. If the value is positive, the IC is attributed to the right foot; if negative, it's attributed to the left foot.

        As a first extension to the original McCamley algorithm, the angular velocity around the anterior-posterior axis, "gyr_z", can resemble a periodic wave with a constant phase shift w.r.t. "gyr_x" after application of the low-pass filter described above. This is also a suitable input signal for the McCamley algorithm, when inverting the sign. A second and final extension to the original McCamley algorithm is to use the combination of the filtered signals for the vertical and anterior-posterior signals: gyr_comb = gyr_x - gyr_z.
    :
        The methodology used here is based on the following reference papers:
        1) J. McCamley et al., “An enhanced estimate of initial contact and final contact instants of time using lower trunk inertial sensor data,” Gait & posture, 2012, available at:
        https://www.sciencedirect.com/science/article/pii/S0966636212000707?via%3Dihub

        2) Reference Papers: Ullrich M, Kuderle A, Reggi L, Cereatti A, Eskofier BM, Kluge F. Machine learning-based distinction of left and right foot contacts in lower back inertial sensor gait data. Annu Int Conf IEEE Eng Med Biol Soc. 2021, available at: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9630653
    """
    axis: Literal["yaw", "roll", "combined"]
    smoothing_filter: BaseFilter

    # TODO: Note that the smoothing filter must filter a DC offset
    smoothed_data_: pd.DataFrame

    def __init__(
        self,
        axis: Literal["yaw", "roll", "combined"] = "combined",
        smoothing_filter: BaseFilter = cf(
            ButterworthFilter(order=4, cutoff_freq_hz=(0.5, 2), filter_type="bandpass")
        ),
    ):
        self.axis = axis
        self.smoothing_filter = smoothing_filter

    @base_lrd_docfiller
    def detect(
        self,
        data: pd.DataFrame,
        ic_list: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        **_: Unpack[dict[str, Any]]
    ) -> Self:
        """%(detect_short)s.

        Parameters
        ----------
        %(detect_para)s

        %(detect_return)s
        """
        self.sampling_rate_hz = sampling_rate_hz
        self.data = data
        self.ic_list = ic_list

        # create a copy of ic_list, otherwise they will get modified when adding the predicted labels
        ic_list = ic_list.copy()

        if self.axis == "yaw":
            selected_data = data["gyr_x"]
        elif self.axis == "roll":
            # roll is phase shifted compared to yaw -> invert the sign
            selected_data = data["gyr_z"] * -1
        elif self.axis == "combined":
            # combine both signals to amplify the differences between left and right steps
            selected_data = data["gyr_z"] * -1 + data["gyr_x"]
        else:
            raise ValueError(f"Invalid axis configuration: {self.axis}.")

        # The use of the smoothing filter is an addition made by Ullrich et al. to the original McCamley algorithm.
        # Originally, simply the mean of the signal was subtracted from the signal
        self.smoothed_data_ = self.smoothing_filter.clone().filter(
            selected_data, sampling_rate_hz=self.sampling_rate_hz
        ).filtered_data_

        data_at_ic = self.smoothed_data_.iloc[ic_list["ic"].to_numpy()]

        ic_list["lr"] = np.where(data_at_ic <= 0, "left", "right")

        self.ic_lr_list_ = ic_list
        return self
