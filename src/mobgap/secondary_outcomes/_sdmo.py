from typing import Any, Optional

import pandas as pd
from scipy.signal import detrend
from typing_extensions import Self, Unpack

from mobgap.secondary_outcomes.base import BaseSDMOCalculator, base_sdmo_docfiller
from mobgap.utils.dtypes import assert_is_sensor_data


@base_sdmo_docfiller
class SDMO(BaseSDMOCalculator):
    r"""Secondary digital mobility outcome calculations on IMU signal (ideally per walking bout).

    This "algorithm" calculates secondary outcomes for given signal window.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(secondary_outcomes)s

    """

    @base_sdmo_docfiller
    def calculate(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """%(calculate_short)s.

        Parameters
        ----------
        %(calculate_para)s
        %(calculate_return)s
        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        # expected the input data in body frame
        assert_is_sensor_data(self.data, frame="body")
        # collect all methods implementing SDMO calculation (add new ones to this list)
        # alternatively, inspect.getmembers can be used to get all methods (such as those starting with "_calculate")
        SDMO_functions = [self._calculate_rms]
        row = {"start": 0, "end": len(data)}
        for func in SDMO_functions:
            row.update(func(data).to_dict())
        self.secondary_outcomes = pd.DataFrame([row])
        return self

    def _calculate_rms(self, data: pd.DataFrame) -> pd.Series:
        """Compute acceleration, gyroscope, total acceleration signal root-mean-square (RMS), and ratio metrics.
        Ratio between RMS of axes i to RMSAccTotal (i = is, ml or pa)
        RMS ratio is based on the following article:
        A gait abnormality measure based on root mean square of trunk acceleration
        Masaki Sekine et al. Journal of NeuroEngineering and Rehabilitation 2013, 10:118
        http://www.jneuroengrehab.com/content/10/1/118
        """
        # first remove DC of acc signals
        data = data.copy()
        data.loc[:, data.columns.str.contains("acc")] = detrend(data.filter(like="acc").to_numpy(), axis=0)
        rms = (data.pow(2).mean() ** 0.5).add_prefix("RMS_")
        # total RMS
        rms_total_acc = ((rms[rms.index.str.contains("acc")]).pow(2).sum()) ** 0.5
        rms["RMSTotal_acc"] = rms_total_acc
        # ratio rms
        for key in [k for k in rms.keys() if k.startswith("RMS_acc") and k != "RMSTotal_acc"]:
            axis = key.replace("RMS_", "")
            rms[f"RMSRatio_{axis}"] = rms[key] / rms_total_acc if rms_total_acc != 0 else 0
        return rms

    def _calculate_regularity_similarity(self, data: pd.DataFrame) -> pd.Series:
        pass

    def _calculate_freq_amp_width_slope(self, data: pd.DataFrame) -> pd.Series:
        pass

    def _calculate_jerk(self, data: pd.DataFrame) -> pd.Series:
        pass

    def _calculate_sd_range(self, data: pd.DataFrame) -> pd.Series:
        pass