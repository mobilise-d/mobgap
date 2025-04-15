"""Base classes for all data transformers and filters."""

from typing import Any, ClassVar, Literal, Optional, Union

import numpy as np
from scipy.signal import filtfilt, lfilter, sosfilt, sosfiltfilt
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc
from mobgap.utils.dtypes import DfLike, dflike_as_2d_array

base_transformer_docfiller = make_filldoc(
    {
        "results": """
        transformed_data_
            The transformed data.
            The datatype matches the datatype of the passed data.
        """,
        "other_parameters": """
        data
            The raw data passed to the ``transform`` method.
            This can either be a dataframe, a series, or a numpy array.
        sampling_rate_hz
            The sampling rate of the IMU data in Hz passed to the ``transform`` method.
        """,
        "transform_short": """
        Transform the passed data.
        """,
        "transform_para": """
        data
            The raw data to be transformed.
            This can either be a dataframe, a series, or a numpy array.
        sampling_rate_hz
            The sampling rate of the IMU data in Hz.
        """,
        "transform_kwargs": """
        kwargs
            Further keyword arguments for the filter.
            They only exist in the base class to allow subclasses to add further parameters.
            However, the base method itself does not use them.
        """,
        "filter_return": """
        Returns
        -------
        self
            The instance of the class with the ``transformed_data_`` attribute set to the filtered
            data.
        """,
    },
    doc_summary="Decorator to fill common parts of the docstring for subclasses of :class:`BaseFilter`.",
)


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

    def _get_updated_chain_kwargs(self, **kwargs: Unpack[dict[str, Any]]) -> dict[str, Any]:
        """Update the kwargs for the next transformer in the chain.

        This method is used to update the kwargs for the next transformer in the chain.
        This is only relevant in combination with the :func:`chain_transformers` function.

        It allows a transformer to update the kwargs passed to the transform method (including the sampling rate) for
        the next transformer in the chain.
        A concrete usecase is the :class:`Resample` transformer, which provides an output with a different sampling rate
        and hence needs to update the sampling rate for the next transformer in the chain.

        This method is always ever called on instances that already have results attached.
        So you can make use of results in the update process.

        By default, this method does nothing and just returns the passed kwargs.
        """
        return kwargs


base_filter_docfiller = make_filldoc(
    {
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
            The instance of the class with the ``transformed_data_``/``filtered_data_`` attribute set to the filtered
            data.
        """,
    },
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
            The data represented either as a dataframe, a series, or a numpy array.
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
        self, data: DfLike, *, sampling_rate_hz: Optional[float] = None, **kwargs: Unpack[dict[str, Any]]
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
        **base_filter_docfiller._dict,
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
        """Get the filter coefficients.

        This should return the filter coefficients in the ``b, a`` format used by :func:`scipy.signal.lfilter` and
        :func:`scipy.signal.filtfilt`.
        The first returned array should contain the ``b`` coefficients and the second array should contain the ``a``
        """
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
        if sampling_rate_hz is None:
            raise ValueError(
                f"{type(self).__name__}.filter requires a `sampling_rate_hz` to be passed. "
                "Currently, `None` (the default value) is passed."
            )

        if sampling_rate_hz != self.EXPECTED_SAMPLING_RATE_HZ:
            raise ValueError(
                f"{type(self).__name__} requires a sampling rate of {self.EXPECTED_SAMPLING_RATE_HZ} Hz. "
                "If your data has a different sampling rate, please resample it first."
            )

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        data, index, transformation_func = dflike_as_2d_array(data)

        filter_func = filtfilt if self.zero_phase else lfilter

        transformed_data = filter_func(*self.coefficients, data, axis=0)

        self.transformed_data_ = transformation_func(transformed_data, index)

        return self


scipy_filter_docfiller = make_filldoc(
    {
        **base_filter_docfiller._dict,
        "common_paras": """
    order
        The filter order.
    cutoff_freq_hz
        The critical frequencies describing the filter.
        This depends on the filter type.
        If the filtertype requires only a single frequence ("lowpass" or "highpass"), this should be a single float.
        If the filtertype requires two frequencies ("bandpass" or "bandstop"), this should be a tuple of two floats.
    filter_type
        The filter type ("lowpass", "highpass", "bandpass", "bandstop").
    """,
        "zero_phase_ba": """
    zero_phase
        Whether to apply a zero-phase filter (i.e. forward and backward filtering) using :func:`scipy.signal.filtfilt`/
        or a normal forward filter using :func:`scipy.signal.lfilter`.
    """,
        "zero_phase_sos": """
    zero_phase
        Whether to apply a zero-phase filter (i.e. forward and backward filtering) using
        :func:`scipy.signal.sosfiltfilt`/ or a normal forward filter using :func:`scipy.signal.sosfilter`.
    """,
        "filter_kwargs": """
    _
        Dummy to catch further parameters.
        They are ignored.
    """,
    },
    doc_summary="Decorator to fill common parts of the docstring for subclasses of :class:`ScipyFilter`.",
)


@scipy_filter_docfiller
class ScipyFilter(BaseFilter):
    """Base class for generic filters using the scipy filter functions.

    Child-classes should specify `_FILTER_TYPE` as class var and implement `_sos_filter_design` or `_ba_filter_design`
    depending on the `_FILTER_TYPE`.

    Parameters
    ----------
    %(common_paras)s
    zero_phase
        Whether to apply a zero-phase filter (i.e. forward and backward filtering) using
        :func:`scipy.signal.sosfiltfilt`/:func:`scipy.signal.filtfilt` or a normal forward filter using
        :func:`scipy.signal.sosfilter`/:`func:`scipy.signal.lfilter`, depending on the `_FILTER_TYPE` of the
        child class.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(results)s

    """

    _FILTER_TYPE: ClassVar[Literal["sos", "ba"]]
    _METHODS: ClassVar = {
        "sos": {"single_pass": sosfilt, "double_pass": sosfiltfilt},
        "ba": {"single_pass": lfilter, "double_pass": filtfilt},
    }

    order: int
    cutoff_freq_hz: Union[float, tuple[float, float]]
    zero_phase: bool
    filter_type: Literal["lowpass", "highpass", "bandpass", "bandstop"]

    sampling_rate_hz: float

    def __init__(
        self,
        order: int,
        cutoff_freq_hz: Union[float, tuple[float, float]],
        *,
        filter_type: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass",
        zero_phase: bool = True,
    ) -> None:
        self.order = order
        self.cutoff_freq_hz = cutoff_freq_hz
        self.zero_phase = zero_phase
        self.filter_type = filter_type

    @scipy_filter_docfiller
    def filter(self, data: DfLike, *, sampling_rate_hz: Optional[float] = None, **_: Unpack[dict[str, Any]]) -> Self:
        """Filter the data.

        This will apply the filter along the **first** axis (axis=0) (aka each column will be filtered).

        Parameters
        ----------
        %(filter_para)s
        %(filter_kwargs)s

        %(filter_return)s

        """
        if sampling_rate_hz is None:
            raise ValueError(
                f"{type(self).__name__}.filter requires a `sampling_rate_hz` to be passed. "
                "Currently, `None` (the default value) is passed."
            )

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        data, index, transformation_func = dflike_as_2d_array(data)
        pass_type = "double_pass" if self.zero_phase else "single_pass"

        if self._FILTER_TYPE == "sos":
            sos = self._sos_filter_design(sampling_rate_hz)
            transformed_data = self._METHODS["sos"][pass_type](sos=sos, x=data, axis=0)
        elif self._FILTER_TYPE == "ba":
            b, a = self._ba_filter_design(sampling_rate_hz)
            transformed_data = self._METHODS["ba"][pass_type](b, a, x=data, axis=0)
        else:
            raise ValueError(f"Unknown filter type: {self._FILTER_TYPE}")

        self.transformed_data_ = transformation_func(transformed_data, index)
        return self

    def _sos_filter_design(self, sampling_rate_hz: float) -> np.ndarray:
        """Design the filter.

        Parameters
        ----------
        sampling_rate_hz
            The sampling rate of the data in Hz

        Returns
        -------
        The filter coefficients in sos format

        """
        raise NotImplementedError()

    def _ba_filter_design(self, sampling_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
        """Design the filter.

        Parameters
        ----------
        sampling_rate_hz
            The sampling rate of the data in Hz

        Returns
        -------
        The filter coefficients in b, a format

        """
        raise NotImplementedError()


class IdentityFilter(BaseFilter):
    """Do nothing.

    Just returns a copy of the input data.
    """

    @scipy_filter_docfiller
    def filter(self, data: DfLike, *, sampling_rate_hz: Optional[float] = None, **_: Unpack[dict[str, Any]]) -> Self:
        """Filter the data by doing absolutely nothing.

        Parameters
        ----------
        %(filter_para)s
        %(filter_kwargs)s

        %(filter_return)s

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.transformed_data_ = data.copy()
        return self


__all__ = [
    "BaseFilter",
    "BaseTransformer",
    "FixedFilter",
    "IdentityFilter",
    "ScipyFilter",
    "base_filter_docfiller",
    "base_transformer_docfiller",
    "fixed_filter_docfiller",
    "scipy_filter_docfiller",
]
