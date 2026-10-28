"""StandardDeviationAndRange calculation for one prepared window."""

from typing import Any

import pandas as pd
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import timed_action_method
from mobgap.secondary_outcomes.base import BaseSecondaryOutcomeCalculator, base_secondary_outcome_docfiller


@base_secondary_outcome_docfiller
class StandardDeviationAndRange(BaseSecondaryOutcomeCalculator):
    """Calculate standard deviation of acceleration and gyroscope signals and range of acceleration signals.

    SD is the sample standard deviation (ddof=1); range is maximum minus minimum. Acceleration outputs are in
    m/s² and gyroscope SD in degrees/s, without hidden unit conversion.

    Other Parameters
    ----------------
    %(data_param)s

    Attributes
    ----------
    %(signal_based_parameters_)s
    %(perf_)s

    """

    @timed_action_method
    @base_secondary_outcome_docfiller
    def calculate(self, data: pd.DataFrame, **_kwargs: Unpack[dict[str, Any]]) -> Self:
        """%(calculate_short)s.

        Parameters
        ----------
        %(data_param)s

        %(calculate_return)s

        """
        self.data = data
        self.signal_based_parameters_ = pd.Series(dtype=float)
        out = {}
        for c in data.columns:
            out[f"sd_{c}"] = data[c].std()
            if "acc" in c:  # range only for the acc columns
                out[f"range_{c}"] = data[c].max() - data[c].min()
        self.signal_based_parameters_ = pd.Series(out, dtype=float)
        return self
