from typing import Union, Unpack, Any, Self, Optional

import numpy as np
import pandas as pd

from gaitlink.data_transform.base import BaseTransformer
from gaitlink.utils.dtypes import DfLike, dflike_as_2d_array


class Cut(BaseTransformer):
    cut_width_s: Union[float, tuple[float, float]]

    sample_rate_hz: float

    def __init__(self, cut_width_s: Union[float, tuple[float, float]]) -> None:
        self.cut_width_s = cut_width_s

    def transform(self, data: DfLike, *, sampling_rate_hz: Optional[float] = None, **_: Unpack[dict[str, Any]]) -> Self:
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        if sampling_rate_hz is None:
            raise ValueError("The sampling rate must be provided.")

        if isinstance(self.cut_width_s, tuple):
            if not len(self.cut_width_s) == 2:
                raise ValueError("If a tuple is given for the `cut_width_s` parameters , it must contain two values.")

            cut_width_samples = tuple(int(np.round(s * sampling_rate_hz)) for s in self.cut_width_s)
        else:
            cut_width_samples = int(np.round(self.cut_width_s * sampling_rate_hz))
            cut_width_samples = (cut_width_samples, cut_width_samples)

        data_as_array, index, transformation_func = dflike_as_2d_array(data)

        if index is not None:
            index = index[cut_width_samples[0]:-cut_width_samples[1]]

        self.transformed_data_ = transformation_func(data_as_array[cut_width_samples[0]:-cut_width_samples[1]], index)

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

    """

    pad_width_s: Union[float, tuple[float, float]]
    mode: str

    sample_rate_hz: float

    pad_width_samples_: Union[int, tuple[int, int]]

    def __init__(self, pad_width_s: Union[float, tuple[float, float]], mode: str = "wrap") -> None:
        self.pad_width_s = pad_width_s
        self.mode = mode

    def transform(self, data: DfLike, *, sampling_rate_hz: Optional[float] = None, **_: Unpack[dict[str, Any]]) -> Self:
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # TODO: Handle the index properly

        if sampling_rate_hz is None:
            raise ValueError("The sampling rate must be provided.")

        if isinstance(self.pad_width_s, tuple):
            if not len(self.pad_width_s) == 2:
                raise ValueError("If a tuple is given for the `pad_with_s` parameters , it must contain two values.")

            self.pad_width_samples_ = tuple(int(np.round(s * sampling_rate_hz)) for s in self.pad_width_s)
            padding_as_tuple = self.pad_width_samples_
        else:
            tmp = int(np.round(self.pad_width_s * sampling_rate_hz))
            self.pad_width_samples_ = tmp
            padding_as_tuple = (tmp, tmp)

        data_as_array, index, transformation_func = dflike_as_2d_array(data)

        padded_data = np.pad(data_as_array, (padding_as_tuple, (0, 0)), mode=self.mode)

        # TODO: Handle the index properly
        self.transformed_data_ = transformation_func(padded_data, None)

        return self

    def get_inverse_transformer(self) -> Cut:
        """Get the inverse transformer for the padding."""
        return Cut(cut_width_s=self.pad_width_s)
