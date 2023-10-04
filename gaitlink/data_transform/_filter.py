from importlib.resources import open_text
from typing import ClassVar

import numpy as np
import pandas as pd

from gaitlink._docutils import inherit_docstring_from
from gaitlink.data_transform.base import FixedFilter, fixed_filter_docfiller


@fixed_filter_docfiller
class EpflGaitFilter(FixedFilter):
    """A filter developed by EPFL to enhance gait related signals in noisy IMU data from lower-back sensors.

    .. warning::
        This filter is only intended to be used with data sampled at 40 Hz.

    Parameters
    ----------
    %(zero_phase)s

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(results)s

    """

    _COEFFS_FILE_NAME: ClassVar[str] = "epfl_gait_filter.csv"
    EXPECTED_SAMPLING_RATE_HZ: ClassVar[float] = 40.0

    @property
    @inherit_docstring_from(FixedFilter)
    def coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        with open_text(
            "gaitlink.data_transform._filter_coeffs",
            self._COEFFS_FILE_NAME,
        ) as test_data:
            coeffs = pd.read_csv(test_data, header=0)["coefficients"].to_numpy()
        return coeffs, np.array(1)

@fixed_filter_docfiller
class EpflDedriftFilter(FixedFilter):
    """A custom IIR filter developed by EPFL to remove baseline drift.

    .. warning::
        This filter is only intended to be used with data sampled at 40 Hz.

    Parameters
    ----------
    %(zero_phase)s

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(results)s

    """

    EXPECTED_SAMPLING_RATE_HZ: ClassVar[float] = 40.0

    @property
    @inherit_docstring_from(FixedFilter)
    def coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([1.0, -1.0]), np.array([1.0, -0.9748])
