"""Utility functions to perform common array operations."""

import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.lib.stride_tricks import sliding_window_view as np_sliding_window_view

from mobgap._gaitmap.utils.array_handling import (
    bool_array_to_start_end_array,
    merge_intervals,
    start_end_array_to_bool_array,
)


def sliding_window_view(data: np.ndarray, window_size_samples: int, overlap_samples: int) -> np.ndarray:
    """Create a sliding window view of the data.

    Note, the output will be a view of the data, not a copy.
    This makes it more efficient, but also means, that the data should not be modified.

    If the length of the data can not be divided by the window size, remaining samples will be dropped.

    Parameters
    ----------
    data
        The data to create the sliding window view of.
        This data can be n-dimensional.
        However, the window will only be applied to the first axis (axis=0).
        See Notes for details.
    window_size_samples
        The size of the sliding window in samples.
    overlap_samples
        The overlap of the sliding window in samples.

    Returns
    -------
    np.ndarray
        The sliding window view of the data.

    Notes
    -----
    In case of nd-arrays, the output format looks as follows:

    Assume the input array has the shape (I, J, K) and we want to create a sliding window view with a window size of N
    and no overlap.
    Then the output array will have the shape (I // N, N, J, K).

    This is different from the output of :func:`numpy.lib.stride_tricks.sliding_window_view`, which would have the shape
    (I // N, J, K, N).
    We changed this here, so that when inspecting each window, you would still have the expected array shape, assuming
    that your original first axis was time.

    """
    if overlap_samples > window_size_samples:
        raise ValueError("overlap_samples must be smaller than window_size_samples")

    # Validate the window on at most one window of input. Constructing every
    # one-sample offset first can exceed np.intp on 32-bit platforms, even when
    # the subsampled result is small enough to represent.
    np_sliding_window_view(data[:window_size_samples], window_shape=(window_size_samples,), axis=0)
    step = window_size_samples - overlap_samples
    n_windows = len(range(0, data.shape[0] - window_size_samples + 1, step))
    return as_strided(
        data,
        shape=(n_windows, window_size_samples, *data.shape[1:]),
        strides=(step * data.strides[0], data.strides[0], *data.strides[1:]),
        writeable=False,
    )


__all__ = ["bool_array_to_start_end_array", "merge_intervals", "sliding_window_view", "start_end_array_to_bool_array"]
