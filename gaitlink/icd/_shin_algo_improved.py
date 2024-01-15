from typing import Any, Literal

import numpy as np
import pandas as pd
from gaitmap.data_transform import Resample
from numpy.linalg import norm
from scipy.ndimage import gaussian_filter
from scipy.signal import cwt, ricker, savgol_filter
from typing_extensions import Self, Unpack

from gaitlink.data_transform import EpflDedriftedGaitFilter, EpflGaitFilter
from gaitlink.icd._utils import zerocros
from gaitlink.icd.base import BaseIcdDetector


class IcdShinImproved(BaseIcdDetector):
    axis: Literal["x", "y", "z", "norm"]

    def __init__(self, axis: Literal["x", "y", "z", "norm"] = "norm"):
        self.axis = axis

    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_: Unpack[dict[str, Any]]) -> Self:
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        self.ic_list_ = pd.DataFrame(
            {
                "ics": _shin_algo_improved(
                    self.data[["acc_x", "acc_y", "acc_z"]], sampling_rate_hz=sampling_rate_hz, axis=self.axis
                )
            }
        )

        return self


def _shin_algo_improved(imu_acc, sampling_rate_hz, axis: Literal["x", "y", "z", "norm"] = "norm"):
    # Check if 'data' is provided, if not, set it to the default 'norm' (all axes of accelerometry)

    if axis == "norm":
        # TODO: Double check correct axis
        accN = norm(imu_acc.to_numpy(), axis=1)
    else:
        accN = imu_acc[axis].to_numpy()

    # Resample to 40Hz to process with filters
    current_sampling_rate = sampling_rate_hz
    target_sampling_rate = 40
    resampler = Resample(target_sampling_rate)
    resampler.transform(data=accN, sampling_rate_hz=current_sampling_rate)
    accN40 = resampler.transformed_data_

    # We need to intitialize the filter once to get the number of coefficients to calculate the padding.
    # This is not ideal, but works for now.
    # TODO: We should evaluate, if we need the padding at all, or if the filter methods that we use handle that
    #  correctly anyway.
    n_coefficients = len(EpflGaitFilter().coefficients[0])

    # Padding to cope with short data
    len_pad = 4 * n_coefficients
    accN40_zp = np.pad(accN40, (len_pad, len_pad), "wrap")

    # TODO: change number on files below
    # Filters
    # 1
    accN_filt1 = savgol_filter(accN40_zp.squeeze(), window_length=21, polyorder=7)
    # 2
    filter = EpflDedriftedGaitFilter()
    # TODO: Why is the output not used?
    accN_filt2 = filter.filter(accN_filt1, sampling_rate_hz=40).filtered_data_
    # 3
    accN_filt3 = cwt(accN_filt1.squeeze(), ricker, [10])
    # 4
    accN_filt4 = savgol_filter(accN_filt3.squeeze(), window_length=11, polyorder=5)
    # 5
    accN_filt5 = cwt(accN_filt4.squeeze(), ricker, [10])
    # 6
    windowWidth = 10
    sigma = windowWidth / 5
    accN_filt6 = gaussian_filter(accN_filt5.squeeze(), sigma)
    # 7
    windowWidth = 10
    sigma = windowWidth / 5
    accN_filt7 = gaussian_filter(accN_filt6.squeeze(), sigma)
    # 8
    windowWidth = 15
    sigma = windowWidth / 5
    accN_filt8 = gaussian_filter(accN_filt7.squeeze(), sigma)
    # MultiFilt
    accN_MultiFilt_rmp = accN_filt8[len_pad:-len_pad]

    # TODO (for future): Upsampling here does not seem that "smart". Maybe better to detect zero crossing in low freq
    #                    and then upsample results. Should be much more efficient.
    # Resample to 100Hz for consistency with the original data (for ICD)
    current_sampling_rate = 40
    resampler = Resample(sampling_rate_hz)
    resampler.transform(data=accN_MultiFilt_rmp, sampling_rate_hz=current_sampling_rate)
    accN_MultiFilt_rmp100 = resampler.transformed_data_

    # Initial contacts timings (heel strike events) detected as positive slopes zero-crossing in sample 120
    IC_lowSNR = zerocros(accN_MultiFilt_rmp100, "p")
    IC_lowSNR = IC_lowSNR[0]
    IC_lowSNR = np.round(IC_lowSNR)

    return IC_lowSNR
