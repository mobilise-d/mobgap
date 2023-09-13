from typing import Optional

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self

from gaitlink.consts import GRAV
from gaitlink.data_transform import BaseFilter, FirFilter


class BaseGsdPostProcessor(Algorithm):
    _action_methods = ("post_process",)

    # Other paramters
    data: pd.DataFrame
    sampling_rate_hz: float

    # results
    processed_gsd_list_: pd.DataFrame

    def post_process(self, data: pd.DataFrame, gsd_list: pd.DataFrame, *, sampling_rate_hz: float, **kwargs) -> Self:
        raise NotImplementedError


class RemoveDuration(BaseGsdPostProcessor):
    def __init__(self, min_duration_s: Optional[float] = 5, max_duration_s: Optional[float] = None) -> None:
        self.min_duration_s = min_duration_s
        self.max_duration_s = max_duration_s

    def post_process(self, data: pd.DataFrame, gsd_list: pd.DataFrame, *, sampling_rate_hz: float, **_) -> Self:
        duration = (gsd_list["end"] - gsd_list["start"]) / sampling_rate_hz
        selected = pd.Series(True, index=gsd_list.index)
        if self.min_duration_s is not None:
            selected &= duration >= self.min_duration_s
        if self.max_duration_s is not None:
            selected &= duration <= self.max_duration_s
        self.processed_gsd_list_ = gsd_list[selected]
        return self


class RemoveNonUpright(BaseGsdPostProcessor):
    def __init__(
        self,
        upright_threshold_ms2: float = 0.5 * GRAV,
        dc_filter: BaseFilter = FirFilter(order=5, cutoff_freq_hz=0.5, filter_type="lowpass"),
    ) -> None:
        self.upright_threshold_ms2 = upright_threshold_ms2
        self.dc_filter = dc_filter

    def post_process(self, data: pd.DataFrame, gsd_list: pd.DataFrame, *, sampling_rate_hz: float, **kwargs) -> Self:
        # We calculate the mean of the vertical acc component for each gsd
        # If the mean is above the threshold, we keep the gsd
        acc_v = self.dc_filter.clone().filter(data["acc_z"], sampling_rate_hz=sampling_rate_hz).transformed_data_
        mean_acc = [acc_v.iloc[start:end].mean() for start, end in gsd_list[["start", "end"]].itertuples(index=False)]
        selected = pd.Series(mean_acc, index=gsd_list.index) > self.upright_threshold_ms2

        self.processed_gsd_list_ = gsd_list[selected]
        return self


class RemoveTransitions(BaseGsdPostProcessor):
    def __init__(self, time_window_s: float = 1, allowed_difference_per: float = 15) -> None:
        self.time_window_s = time_window_s
        self.allowed_difference_per = allowed_difference_per

    def post_process(self, data: pd.DataFrame, gsd_list: pd.DataFrame, *, sampling_rate_hz: float, **kwargs) -> Self:
        # We calculate the mean of the vertical acc component for the first and the last n seconds of each gsd
        # If the difference is above the threshold, we remove the gsd
        acc_v = data["acc_z"]
        for start, end in gsd_list[["start", "end"]].itertuples(index=False):
            acc_v.iloc[start:end]
