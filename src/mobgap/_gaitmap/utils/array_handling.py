from typing import Optional

import numba
import numpy as np
from numba import njit


def merge_intervals(input_array: np.ndarray, gap_size: int = 0) -> np.ndarray:
    """Merge intervals that are overlapping and that are a distance less or equal to gap_size from each other.

    This is actually a wrapper for _solve_overlap that is needed because numba can not compile np.sort().

    Parameters
    ----------
    input_array : (n, 2) np.ndarray
        The np.ndarray containing the intervals that should be merged
    gap_size : int
        Integer that sets the allowed gap between intervals.
        For examples see below.
        Default is 0.

    Returns
    -------
    merged intervals array
        (n, 2) np.ndarray containing the merged intervals

    Examples
    --------
    >>> test = np.array([[1, 3], [2, 4], [6, 8], [5, 7], [10, 12], [11, 15], [18, 20]])
    >>> merge_intervals(test)
    array([[ 1,  4],
           [ 5,  8],
           [10, 15],
           [18, 20]])

    >>> merge_intervals(test, 2)
    array([[ 1, 15],
           [18, 20]])

    """
    if input_array.shape[0] == 0:
        return input_array

    return np.array(_solve_overlap(np.sort(input_array, axis=0, kind="stable"), gap_size))


@njit
def _solve_overlap(input_array: np.ndarray, gap_size: int) -> numba.typed.List:
    """Merge intervals that are overlapping and that are a distance less or equal to gap_size from each other."""
    stack = numba.typed.List()
    stack.append(input_array[0])

    for i in range(1, len(input_array)):
        if stack[-1][0] <= input_array[i][0] <= (stack[-1][1] + gap_size) <= (input_array[i][1] + gap_size):
            stack[-1][1] = input_array[i][1]
        else:
            stack.append(input_array[i])

    return stack


def bool_array_to_start_end_array(bool_array: np.ndarray) -> np.ndarray:
    """Find regions in bool array and convert those to start-end indices.

    The end index is the first element after the True-region,
    so you can use it for upper-bound exclusive slicing etc.

    Parameters
    ----------
    bool_array : array with shape (n,)
        boolean array with either 0/1, 0.0/1.0 or True/False elements

    Returns
    -------
    array of [start, end] indices with shape (n,2)

    Examples
    --------
    >>> example_array = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1])
    >>> start_end_list = bool_array_to_start_end_array(example_array)
    >>> start_end_list
    array([[2, 4],
           [6, 9]])
    >>> example_array[start_end_list[0, 0] : start_end_list[0, 1]]
    array([1, 1])

    """
    # check if input is a numpy array. E.g. for a Series, start and end will just be the first and last index.
    if not isinstance(bool_array, np.ndarray):
        raise TypeError("Input must be a numpy array!")

    # check if input is actually a boolean array
    if not np.array_equal(bool_array, bool_array.astype(bool)):
        raise ValueError("Input must be boolean array!")

    if len(bool_array) == 0:
        return np.array([])

    slices = np.ma.flatnotmasked_contiguous(np.ma.masked_equal(bool_array, 0))
    return np.array([[s.start, s.stop] for s in slices])


def start_end_array_to_bool_array(start_end_array: np.ndarray, pad_to_length: Optional[int] = None) -> np.ndarray:
    """Convert a start-end list to a bool array.

    Parameters
    ----------
    start_end_array : array with shape (n,2)
        2d-array indicating start and end values e.g. [[10,20],[20,40],[70,80]]

        .. warning:: We assume that the end index is exclusiv!
                     This is in line with the definitions of stride and roi lists in gaitmap.

    pad_to_length: int
        Define the length of the resulting array.
        If None, the array will have the length of the largest index.
        Otherwise, the final array will either be padded with False or truncated to the specified length.

    Returns
    -------
    array with shape (n,)
        boolean array with True/False elements

    Examples
    --------
    >>> import numpy as np
    >>> example_array = np.array([[3, 5], [7, 8]])
    >>> start_end_array_to_bool_array(example_array, pad_to_length=12)
    array([False, False, False,  True,  True,  True, False,  True,  True,
           False, False, False])
    >>> example_array = np.array([[3, 5], [7, 8]])
    >>> start_end_array_to_bool_array(example_array, pad_to_length=None)
    array([False, False, False,  True,  True,  True, False,  True,  True])

    """
    start_end_array = np.atleast_2d(start_end_array)

    if pad_to_length is None:
        n_elements = start_end_array.max()
    else:
        if pad_to_length < 0:
            raise ValueError("pad_to_length must be positive!")
        n_elements = pad_to_length

    bool_array = np.zeros(n_elements)
    for start, end in start_end_array:
        bool_array[start:end] = 1
    return bool_array.astype(bool)
