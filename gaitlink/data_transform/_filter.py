from importlib.resources import files
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd
from typing_extensions import Self, Unpack

from gaitlink._docutils import inherit_docstring_from
from gaitlink.data_transform.base import BaseFilter, FixedFilter, chain_transformers, fixed_filter_docfiller
from gaitlink.utils.dtypes import DfLike


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
        with (files("gaitlink") / "data_transform/_filter_coeffs" / self._COEFFS_FILE_NAME).open() as test_data:
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


@fixed_filter_docfiller
class EpflDedriftedGaitFilter(BaseFilter):
    """A filter combining the :class:`EpflDedriftFilter` and :class:`EpflGaitFilter`.

    This filter exists, as these two filters are often used together.
    It just provides a convenient wrapper without any further optimization.
    The dedrifting filter is applied first and then the gait filter.
    I.e. it is equivalent to the following code:

    .. code-block:: python

        dedrifted_data = EpflDedriftFilter().filter(data, sampling_rate_hz=40.0).filtered_data_
        result = EpflGaitFilter().filter(dedrifted_data, sampling_rate_hz=40.0)

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

    EXPECTED_SAMPLING_RATE_HZ: ClassVar[float]

    zero_phase: bool

    def __init__(self, zero_phase: bool = True) -> None:
        self.zero_phase = zero_phase

    @fixed_filter_docfiller
    def filter(self, data: DfLike, *, sampling_rate_hz: Optional[float] = None, **_: Unpack[dict[str, Any]]) -> Self:
        """%(filter_short)s.

        Note, that the sampling rate will not change the filter coefficients.
        Instead, the sampling rate is only used to check, that the passed data has the expected sampling rate.
        If not, a ValueError is raised.
        Hence, the ``sampling_rate_hz`` parameter only exists to make sure that you are reminded of the expected
        sampling rate.

        Parameters
        ----------
        %(filter_para)s
        %(filter_kwargs)s

        %(filter_return)s
        """
        filter_chain = [
            ("dedrift", EpflDedriftFilter(zero_phase=self.zero_phase)),
            ("gait_filter", EpflGaitFilter(zero_phase=self.zero_phase)),
        ]
        self.transformed_data_ = chain_transformers(data, filter_chain, sampling_rate_hz=sampling_rate_hz)

        return self
