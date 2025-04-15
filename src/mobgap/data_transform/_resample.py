import warnings
from typing import Any, Optional

import pandas as pd
from pandas.core.dtypes.common import is_datetime64_any_dtype, is_numeric_dtype
from scipy import signal
from tpcp import clone
from typing_extensions import Self, Unpack

from mobgap.data_transform.base import BaseTransformer
from mobgap.utils.dtypes import DfLike, dflike_as_2d_array


class Resample(BaseTransformer):
    """Resample the input data to a specified target sampling rate using :func:`scipy.signal.resample`.

    Parameters
    ----------
    target_sampling_rate_hz
        The target sampling rate in Hertz.
        If the target sampling rate is equal to the sampling rate of the input data, no resampling is performed.
    attempt_index_resample
        Whether to attempt to resample the index of the input data.
        This is only used if the input data is a DataFrame or Series with a numeric index.
        In this case we assume that the index represents the time or the samples, and we try to resample it.
        If the index is neither numeric nor a datetime objects, we can not resample it and this parameter is ignored.
        In case you index does not represent the time (either in actual time or samples), you should set this parameter
        to False.

    Attributes
    ----------
    transformed_data_
        The resampled data

    Other Parameters
    ----------------
    data
        The data represented either as a dataframe, a series, or a numpy array.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz.

    """

    target_sampling_rate_hz: float
    attempt_index_resample: bool

    sampling_rate_hz: float

    def __init__(self, target_sampling_rate_hz: float = 100.0, *, attempt_index_resample: bool = True) -> None:
        self.target_sampling_rate_hz = target_sampling_rate_hz
        self.attempt_index_resample = attempt_index_resample

    def transform(self, data: DfLike, *, sampling_rate_hz: Optional[float] = None, **_: Unpack[dict[str, Any]]) -> Self:
        """Resample the input data to the target sampling rate.

        Parameters
        ----------
        data
            The data represented either as a dataframe, a series, or a numpy array.
        sampling_rate_hz
            The sampling rate of the IMU data in Hz.

        Returns
        -------
        Resample
            The instance of the transform with the results attached

        Raises
        ------
        ValueError
            If sampling_rate_hz is None.

        """
        if sampling_rate_hz is None:
            raise ValueError("Parameter 'sampling_rate_hz' must be provided.")

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        if sampling_rate_hz == self.target_sampling_rate_hz:
            # No need to resample if the sampling rates match
            self.transformed_data_ = clone(data)  # clone effectively deep copies the data

        # This converts DataFrames, Series, or arrays all into a unified shaped np.ndarray
        data_as_array, index, transformation_func = dflike_as_2d_array(data)

        if self.attempt_index_resample is False:
            index = None

        if index is not None and not (is_numeric_dtype(index) or is_datetime64_any_dtype(index)):
            warnings.warn(
                "You passed an object with an index. "
                "However, the index is not Numeric or a Datetime object. "
                "Hence, we can not resample it."
                "The index will be ignored during resampling.\n\n"
                "You can silence this warning by setting `attempt_index_resample=False`.",
                stacklevel=2,
            )
            index = None

        resampling_factor = self.target_sampling_rate_hz / sampling_rate_hz
        new_n_samples = round(len(data_as_array) * resampling_factor)

        # If we don't have an index (i.e. when a np.array was passed), we don't need to resample the index
        if index is None:
            resampled_data = signal.resample(data, new_n_samples)
        else:
            # Otherwise, we also try to resample the index
            resampled_data, resampled_index = signal.resample(data, new_n_samples, t=index.to_numpy())
            index = pd.Index(resampled_index, name=index.name)

        self.transformed_data_ = transformation_func(resampled_data, index)

        return self

    def _get_updated_chain_kwargs(self, **kwargs: Unpack[dict[str, Any]]) -> dict[str, Any]:
        """Update the chain kwargs with the target sampling rate for the next transformer."""
        return {**kwargs, "sampling_rate_hz": self.target_sampling_rate_hz}
