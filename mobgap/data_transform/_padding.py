import warnings
from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
from typing_extensions import Self, Unpack

from mobgap.data_transform.base import BaseTransformer, base_transformer_docfiller
from mobgap.utils.conversions import as_samples
from mobgap.utils.dtypes import DfLike, dflike_as_2d_array


@base_transformer_docfiller
class Crop(BaseTransformer):
    """Crop the input data to by removing a specified amount of samples from the beginning and end of the data.

    Parameters
    ----------
    crop_len_s
        The length of the data to crop in seconds. If a single value is given, the same amount is cropped from the
        beginning and end of the data.
        If a tuple is given, the first value is used for the beginning and the second for the end.
        The value is converted to samples using the sampling rate of the data.

    Attributes
    ----------
    %(results)s
    crop_len_samples_
        The calculated crop len in samples as calculated from the provided crop_len_s and the sampling rate.

    Other Parameters
    ----------------
    %(other_parameters)s

    """

    crop_len_s: Union[float, tuple[float, float]]

    sampling_rate_hz: float

    crop_len_samples_: Union[int, tuple[int, int]]

    def __init__(self, crop_len_s: Union[float, tuple[float, float]]) -> None:
        self.crop_len_s = crop_len_s

    def transform(self, data: DfLike, *, sampling_rate_hz: Optional[float] = None, **_: Unpack[dict[str, Any]]) -> Self:
        """%(transform_short)s.

        Parameters
        ----------
        %(transform_para)s

        %(transform_return)s
        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        if sampling_rate_hz is None:
            raise ValueError("The sampling rate must be provided.")

        if isinstance(self.crop_len_s, tuple) and len(self.crop_len_s) != 2:
            raise ValueError("If a tuple is given for the `cut_with_s` parameters, it must contain two values.")

        self.crop_len_samples_ = as_samples(self.crop_len_s, sampling_rate_hz)

        crop_len_as_tuple = (
            (self.crop_len_samples_, self.crop_len_samples_)
            if isinstance(self.crop_len_samples_, int)
            else self.crop_len_samples_
        )

        data_as_array, index, transformation_func = dflike_as_2d_array(data)

        if sum(crop_len_as_tuple) > len(data_as_array):
            raise ValueError(
                "The combined crop len is larger than the data. "
                f"crop len: {crop_len_as_tuple} > data length: {len(data_as_array)}"
            )

        if index is not None:
            index = index[crop_len_as_tuple[0] : -crop_len_as_tuple[1]]

        self.transformed_data_ = transformation_func(data_as_array[crop_len_as_tuple[0] : -crop_len_as_tuple[1]], index)

        return self


@base_transformer_docfiller
class Pad(BaseTransformer):
    """Pad the input data using various padding strategies.

    Under the hood we use the `numpy.pad` function to pad the data.

    Parameters
    ----------
    pad_len_s
        Padding len in seconds. If a single value is given, the same padding is applied to the beginning and end of
        the data.
        If a tuple is given, the first value is used for the beginning and the second for the end.
        The value is converted to samples using the sampling rate of the data.
    mode
        Padding mode. See `numpy.pad` for more information.
    constant_values
        The constant value to use for padding in case mode is `constant`. See `numpy.pad` for more information.

    Attributes
    ----------
    %(results)s
    pad_len_samples_
        The calculated padding len in samples as calculated from the provided pad_len_s and the sampling rate.

    Other Parameters
    ----------------
    %(other_parameters)s

    Notes
    -----
    We don't yet support all padding modes.
    Please open an issue if you need support for a specific mode.

    """

    pad_len_s: Union[float, tuple[float, float]]
    mode: str
    constant_values: Union[float, Sequence[float]]

    sampling_rate_hz: float

    pad_len_samples_: Union[int, tuple[int, int]]

    def __init__(
        self,
        pad_len_s: Union[float, tuple[float, float]],
        *,
        mode: str = "reflect",
        constant_values: Optional[Union[float, Sequence[float]]] = None,
    ) -> None:
        self.pad_len_s = pad_len_s
        self.mode = mode
        self.constant_values = constant_values

    def transform(self, data: DfLike, *, sampling_rate_hz: Optional[float] = None, **_: Unpack[dict[str, Any]]) -> Self:
        """%(transform_short)s.

        Parameters
        ----------
        %(transform_para)s

        %(transform_return)s
        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        if sampling_rate_hz is None:
            raise ValueError("The sampling rate must be provided.")

        if isinstance(self.pad_len_s, tuple) and len(self.pad_len_s) != 2:
            raise ValueError("If a tuple is given for the `pad_with_s` parameters, it must contain two values.")

        self.pad_len_samples_ = as_samples(self.pad_len_s, sampling_rate_hz)

        padding_as_tuple = (
            (self.pad_len_samples_, self.pad_len_samples_)
            if isinstance(self.pad_len_samples_, int)
            else self.pad_len_samples_
        )

        data_as_array, index, transformation_func = dflike_as_2d_array(data)

        additional_kwargs = {}
        if self.mode == "constant":
            additional_kwargs["constant_values"] = self.constant_values

        padded_data = np.pad(data_as_array, (padding_as_tuple, (0, 0)), mode=self.mode, **additional_kwargs)

        # TODO: Handle the index properly
        if index is not None:
            warnings.warn(
                "Padding does not yet handle the index properly. "
                "We will ignore the index for now. "
                "This means that the index of the transformed data will be new numeric index starting from 0.",
                stacklevel=1,
            )
        self.transformed_data_ = transformation_func(padded_data, None)

        return self

    def get_inverse_transformer(self) -> Crop:
        """Get the inverse transformer for the padding.

        This returns a `Crop` transformer that can be used to crop the data back to its original size.
        """
        return Crop(crop_len_s=self.pad_len_s)
