from typing import Any, Optional

import pandas as pd
from typing_extensions import Self, Unpack

from mobgap.walking_speed.base import BaseWsCalculator, base_ws_docfiller


@base_ws_docfiller
class WsNaive(BaseWsCalculator):
    r"""Naive walking speed calculation from cadence and stride length.

    This "algorithm" calculates the walking speed from the cadence and stride length, ignoring any other factors.
    The internal equation looks as follows:

    .. math::
        \text{{walking speed}} = \frac{{\text{{stride length}} * \text{{cadence}}}}{60 * 2}

    We assume that the stride length is in meters and the cadence in steps per minute.
    Then we divide by 60 to get steps per second and by 2 to convert from steps per sec to strides per sec.
    Then we multiply by the stride length, getting the walking speed in meters per second.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(walking_speed_per_sec_)s

    """

    @base_ws_docfiller
    def calculate(
        self,
        data: pd.DataFrame,
        *,
        initial_contacts: Optional[pd.DataFrame] = None,
        cadence_per_sec: Optional[pd.DataFrame] = None,
        stride_length_per_sec: Optional[pd.DataFrame] = None,
        sampling_rate_hz: float,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """%(calculate_short)s.

        Parameters
        ----------
        %(calculate_para)s
        %(calculate_return)s
        """
        if cadence_per_sec is None:
            raise ValueError("cadence_per_sec must be provided")
        if stride_length_per_sec is None:
            raise ValueError("stride_length_per_sec must be provided")

        self.data = data
        self.initial_contacts = initial_contacts
        self.sampling_rate_hz = sampling_rate_hz
        self.cadence_per_sec = cadence_per_sec
        self.stride_length_per_sec = stride_length_per_sec

        self.walking_speed_per_sec_ = (
            self.stride_length_per_sec["stride_length_m"] * self.cadence_per_sec["cadence_spm"] / (60 * 2)
        ).to_frame("walking_speed_mps")
        return self
