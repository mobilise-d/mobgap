import warnings
from collections.abc import Sequence
from typing import Any, Optional, Self, Union, Unpack

import numpy as np

from gaitlink.data_transform.base import BaseTransformer
from gaitlink.utils.conversions import as_samples
from gaitlink.utils.dtypes import DfLike, dflike_as_2d_array


class Cut(BaseTransformer):
    cut_width_s: Union[float, tuple[float, float]]

    sampling_rate_hz: float

    cut_width_samples_: Union[int, tuple[int, int]]

    def __init__(self, cut_width_s: Union[float, tuple[float, float]]) -> None:
        self.cut_width_s = cut_width_s

    def transform(self, data: DfLike, *, sampling_rate_hz: Optional[float] = None, **_: Unpack[dict[str, Any]]) -> Self:
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        if sampling_rate_hz is None:
            raise ValueError("The sampling rate must be provided.")

        if isinstance(self.cut_width_s, tuple) and len(self.cut_width_s) != 2:
            raise ValueError("If a tuple is given for the `cut_with_s` parameters , it must contain two values.")

        self.cut_width_samples_ = as_samples(self.cut_width_s, sampling_rate_hz)

        cut_width_as_tuple = (
            (self.cut_width_samples_, self.cut_width_samples_)
            if isinstance(self.cut_width_samples_, int)
            else self.cut_width_samples_
        )

        data_as_array, index, transformation_func = dflike_as_2d_array(data)

        if index is not None:
            index = index[cut_width_as_tuple[0] : -cut_width_as_tuple[1]]

        self.transformed_data_ = transformation_func(
            data_as_array[cut_width_as_tuple[0] : -cut_width_as_tuple[1]], index
        )

        return self


class Pad(BaseTransformer):
    """Pad the input data using various padding strategies.

    Under the hood we use the `numpy.pad` function to pad the data.

    Parameters
    ----------
    pad_width_s
        Padding width in seconds. If a single value is given, the same padding is applied to the beginning and end of
        the data.
        If a tuple is given, the first value is used for the beginning and the second for the end.
        The value is converted to samples using the sampling rate of the data.
    mode
        Padding mode. See `numpy.pad` for more information.
    constant_values
        The constant value to use for padding in case mode is `constant`. See `numpy.pad` for more information.

    Notes
    -----
    We don't yet support all padding modes.
    Please open an issue if you need support for a specific mode.

    """

    pad_width_s: Union[float, tuple[float, float]]
    mode: str
    constant_values: Union[float, Sequence[float]]

    sampling_rate_hz: float

    pad_width_samples_: Union[int, tuple[int, int]]

    def __init__(
        self,
        pad_width_s: Union[float, tuple[float, float]],
        *,
        mode: str = "wrap",
        constant_values: Union[float, Sequence[float]] = None,
    ) -> None:
        self.pad_width_s = pad_width_s
        self.mode = mode
        self.constant_values = constant_values

    def transform(self, data: DfLike, *, sampling_rate_hz: Optional[float] = None, **_: Unpack[dict[str, Any]]) -> Self:
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # TODO: Handle the index properly

        if sampling_rate_hz is None:
            raise ValueError("The sampling rate must be provided.")

        if isinstance(self.pad_width_s, tuple) and len(self.pad_width_s) != 2:
            raise ValueError("If a tuple is given for the `pad_with_s` parameters , it must contain two values.")

        self.pad_width_samples_ = as_samples(self.pad_width_s, sampling_rate_hz)

        padding_as_tuple = (
            (self.pad_width_samples_, self.pad_width_samples_)
            if isinstance(self.pad_width_samples_, int)
            else self.pad_width_samples_
        )

        data_as_array, index, transformation_func = dflike_as_2d_array(data)

        padded_data = np.pad(
            data_as_array, (padding_as_tuple, (0, 0)), mode=self.mode, constant_values=self.constant_values
        )

        # TODO: Handle the index properly\
        if index is not None:
            warnings.warn(
                "Padding does not yet handle the index properly. "
                "We will ignore the index for now. "
                "This means that the index of the transformed data will be new numeric index starting from 0."
            )
        self.transformed_data_ = transformation_func(padded_data, None)

        return self

    def get_inverse_transformer(self) -> Cut:
        """Get the inverse transformer for the padding."""
        return Cut(cut_width_s=self.pad_width_s)
