from typing import Any

import numpy as np
import pandas as pd
from tpcp import cf
from typing_extensions import Self, Unpack

from mobgap.data_transform import ButterworthFilter
from mobgap.data_transform.base import BaseFilter
from mobgap.laterality.base import BaseLRClassifier, _unify_ic_lr_list_df, base_lrc_docfiller
from mobgap.utils.dtypes import assert_is_sensor_data


@base_lrc_docfiller
class LrcMansour(BaseLRClassifier):
    """LrcMansour algorithm for laterality detection of initial contacts.

    The Mansour algorithm [1]_ uses the derivative of the mediolateral acceleration in order to classify left/right
    initial contacts.

    The first step is to filter the signal using a low-pass fourth-order zero-lag Butterworth filter at 1 Hz.

    Second, the derivative of the filtered signal is obtained. This should be approximately sinusoidal.
    An initial contact during a positive derivative is taken as a left initial contact, a negative
    derivative as a right initial contact.

    Parameters
    ----------
    smoothing_filter
        The filter to use for smoothing the signal. The filter should be a low-pass filter to obtain a
        low frequency sinusoidal signal from the mediolateral acceleration.

    Attributes
    ----------
    %(ic_lr_list_)s
    smoothed_data_
        The smoothed mediolateral acceleration used for the laterality detection.
        This might be helpful for debugging or further analysis.

    Other Parameters
    ----------------
    %(other_parameters)s

    Notes
    -----
    In the edge case of data == 0, the right side is assumed.


    .. [1] Mansour, Khaireddine Ben, Nasser Rezzoug, and Philippe Gorce. "Foot side detection from lower lumbar spine
        acceleration." Gait & Posture 42.3 (2015): 386-389., available at:
        https://doi.org/10.1016/j.gaitpost.2015.05.021

    """

    smoothing_filter: BaseFilter

    smoothed_data_: pd.Series

    def __init__(
        self,
        smoothing_filter: BaseFilter = cf(ButterworthFilter(order=4, cutoff_freq_hz=1, filter_type="lowpass")),
    ) -> None:
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

        # Select mediolateral acceleration
        selected_data = data["acc_ml"]

        # Low-pass the acceleration
        self.smoothed_data_ = (
            self.smoothing_filter.clone().filter(selected_data, sampling_rate_hz=self.sampling_rate_hz).filtered_data_
        )

        # Take the derivative
        derivative_data = np.gradient(self.smoothed_data_)

        data_at_ic = derivative_data[ic_list["ic"].to_numpy()]

        ic_list["lr_label"] = np.where(data_at_ic <= 0, "right", "left")

        self.ic_lr_list_ = _unify_ic_lr_list_df(ic_list)
        return self
