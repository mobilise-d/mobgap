from typing import Any, Optional

import pandas as pd
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
        # collect all methods implementing calculation procedures, and add new implementations
        # alternatively, inspect.getmembers can be used to get all methods (such as those starting with "_calculate")
        # below is the list that will be implemented
        SDMO_functions = [self._calculate_rms, self._calculate_regularity_similarity, self._calculate_freq_amp_width_slope,
                          self._calculate_jerk, self._calculate_sd_range]
        row = {"start":0, "end": len(data)}
        for func in SDMO_functions:
            row.update(func(data).to_dict())
        self.secondary_outcomes = pd.DataFrame([row])
        return self

    def _calculate_rms(self, data: pd.DataFrame) -> pd.Series:
        pass

    def _calculate_regularity_similarity(self, data: pd.DataFrame) -> pd.Series:
        pass

    def _calculate_freq_amp_width_slope(self, data: pd.DataFrame) -> pd.Series:
        pass

    def _calculate_jerk(self, data: pd.DataFrame) -> pd.Series:
        pass

    def _calculate_sd_range(self, data: pd.DataFrame) -> pd.Series:
        pass