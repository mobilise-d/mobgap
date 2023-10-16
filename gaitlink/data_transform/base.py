"""Base classes for all data transformers and filters."""
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd
from scipy.signal import filtfilt, lfilter
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from gaitlink._docutils import make_filldoc
from gaitlink.utils.dtypes import DfLike, dflike_as_2d_array


class BaseTransformer(Algorithm):
    """Base class for all data transformers."""

    _action_methods = ("transform",)

    transformed_data_: DfLike

    data: DfLike

    def transform(self, data: DfLike, **kwargs: Unpack[dict[str, Any]]) -> Self:
        """Transform the data using the transformer.

        Parameters
        ----------
        data
            A dataframe representing single sensor data.
        kwargs
            Further keyword arguments for the transformer.

        Returns
        -------
        self
            The instance of the transformer with the results attached

        """
        raise NotImplementedError()


_base_filter_doc_replace = {
    "results": """
    transformed_data_
        The filtered data.
        The datatype matches the datatype of the passed data.
    filtered_data_
        Alias for ``transformed_data_``.
    """,
    "other_parameters": """
    data
        The raw data passed to the ``filter``/``transform`` method.
        This can either be a dataframe, a series, or a numpy array.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz passed to the ``filter``/``transform`` method.
    """,
    "filter_short": """
    Filter the passed data.
    """,
    "filter_para": """
    data
        The raw data to be filtered.
        This can either be a dataframe, a series, or a numpy array.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz.
    """,
    "filter_kwargs": """
    kwargs
        Further keyword arguments for the filter.
        They only exist in the base class to allow subclasses to add further parameters.
        However, the base method itself does not use them.
    """,
    "filter_return": """
    Returns
    -------
    self
        The instance of the class with the ``transformed_data_``/``filtered_data_`` attribute set to the filtered data.
    """,
}

base_filter_docfiller = make_filldoc(
    _base_filter_doc_replace,
    doc_summary="Decorator to fill common parts of the docstring for subclasses of :class:`BaseFilter`.",
)


@base_filter_docfiller
class BaseFilter(BaseTransformer):
    """Base class for all filters.

    This base class should be used for all filters.
    Filters should implement the ``filter`` method, which will perform all relevant processing steps.
    The method should then return the instance of the class, with the ``transformed_data_`` attribute set to the
    filtered data.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(results)s

    Notes
    -----
    You can use the :func:`~base_filter_docfiller` decorator to fill common parts of the docstring for your subclass.
    See the source of this class for an example.

    """

    _action_methods = (*BaseTransformer._action_methods, "filter")

    sampling_rate_hz: float

    @property
    def filtered_data_(self) -> DfLike:
        """Get filtered data.

        This is the same as `transformed_data_` and is just here, as it is easier to remember.
        """
        return self.transformed_data_

    def transform(
        self, data: DfLike, *, sampling_rate_hz: Optional[float] = None, **kwargs: Unpack[dict[str, Any]]
    ) -> Self:
        """Transform the data using the filter.

        This just calls ``self.filter``.
        This method only exists to fulfill the :class:`BaseTransformer` interface.

        Parameters
        ----------
        data
            A dataframe representing single sensor data.
        sampling_rate_hz
            The sampling rate of the IMU data in Hz.
        kwargs
            Further keyword arguments for the filter.

        Returns
        -------
        self
            The instance of the filter with the results attached

        """
        return self.filter(data, sampling_rate_hz=sampling_rate_hz, **kwargs)

    @base_filter_docfiller
    def filter(
        self, data: pd.DataFrame, *, sampling_rate_hz: Optional[float] = None, **kwargs: Unpack[dict[str, Any]]
    ) -> Self:
        """%(filter_short)s.

        Parameters
        ----------
        %(filter_para)s
        %(filter_kwargs)s

        %(filter_return)s


        """
        raise NotImplementedError()


fixed_filter_docfiller = make_filldoc(
    {
        **_base_filter_doc_replace,
        "zero_phase": """
    zero_phase
        Whether to apply a zero-phase filter (i.e. forward and backward filtering) using :func:`scipy.signal.filtfilt`
        or a normal forward filter using :func:`scipy.signal.lfilter`.
    """,
        "EXPECTED_SAMPLING_RATE_HZ": """
    EXPECTED_SAMPLING_RATE_HZ
        (Class Constant) The expected sampling rate of the data in Hz.
        The ``filter`` method will raise a :class:`ValueError` if the passed sampling rate does not match the expected
        one.
    """,
        "filter_kwargs": """
    _
        Dummy to catch further parameters.
        They are ignored.
    """,
    },
    doc_summary="Decorator to fill common parts of the docstring for subclasses of :class:`FixedFilter`.",
)


@fixed_filter_docfiller
class FixedFilter(BaseFilter):
    """Base class for filters with fixed coefficients designed using the typical "ba" parameter format.

    As a filter with fixed coefficients only has the expected properties (i.e. the correct cutoff frequency), at a
    specific sampling rate, this class requires all child-classes to define the expected sampling rate and checks
    explicitly that the passed sampling rate matches the expected one.

    Parameters
    ----------
    %(zero_phase)s

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(results)s
    %(EXPECTED_SAMPLING_RATE_HZ)s

    Notes
    -----
    You can use the :func:`~fixed_filter_docfiller` decorator to fill common parts of the docstring for your subclass.
    See the source of this class for an example.

    """

    EXPECTED_SAMPLING_RATE_HZ: ClassVar[float]

    zero_phase: bool

    def __init__(self, zero_phase: bool = True) -> None:
        self.zero_phase = zero_phase

    @property
    def coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the filter coefficients."""
        raise NotImplementedError()

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
        if sampling_rate_hz != self.EXPECTED_SAMPLING_RATE_HZ:
            raise ValueError(
                f"{type(self).__name__} requires a sampling rate of {self.EXPECTED_SAMPLING_RATE_HZ} Hz. "
                f"If your data has a different sampling rate, please resample it first."
            )

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        data, transformation_func = dflike_as_2d_array(data)

        filter_func = filtfilt if self.zero_phase else lfilter

        transformed_data = filter_func(*self.coefficients, data, axis=0)

        self.transformed_data_ = transformation_func(transformed_data)

        return self


__all__ = ["BaseTransformer", "BaseFilter", "FixedFilter", "fixed_filter_docfiller"]
