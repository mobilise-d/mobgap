"""Some basic conversion utilities."""

from collections.abc import Sequence
from typing import TypeVar, Union, overload

import numpy as np

T = TypeVar("T", float, int, Sequence[float], Sequence[int])


@overload
def as_samples(sec_value: Union[int, float], sampling_rate_hz: float) -> int: ...


@overload
def as_samples(sec_value: Union[Sequence[int], Sequence[float]], sampling_rate_hz: float) -> Sequence[int]: ...


@overload
def as_samples(sec_value: np.ndarray, sampling_rate_hz: float) -> np.ndarray: ...


def as_samples(sec_value, sampling_rate_hz: float):
    """Convert seconds to samples.

    Parameters
    ----------
    sec_value
        The value in seconds.
    sampling_rate_hz
        The sampling rate in Hertz.

    Returns
    -------
    int or Sequence[int]
        The value in samples.

    """
    if isinstance(sec_value, np.ndarray):
        return np.round(sec_value * sampling_rate_hz).astype(int)
    if isinstance(sec_value, (int, float)):
        return int(np.round(sec_value * sampling_rate_hz))
    return type(sec_value)(int(np.round(s * sampling_rate_hz)) for s in sec_value)


__all__ = ["as_samples"]
