"""Contains the groupfind function for group finding in boolean arrays."""
import numpy as np


def groupfind(bool_array: np.ndarray) -> np.ndarray:
    """Find sequences in a bool array which are TRUE/1 and converts those to start-end indices.

    At least 2 consecutive True elements are required to be considered a sequence.
    The end index is the last element of the True-region

    Parameters
    ----------
    bool_array : array with shape (n,)
        boolean array with either 0/1, 0.0/1.0 or True/False elements

    Returns
    -------
    array of [start, end] indices with shape (n,2)

    Examples
    --------
    >>> example_array = np.array([0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0])
    >>> start_end_list = groupfind(example_array)
    >>> start_end_list
    array[[ 5  6]
        [ 9 11]]
    """
    if not isinstance(bool_array, np.ndarray):
        raise TypeError("Input must be a numpy array")

    if not np.array_equal(bool_array, bool_array.astype(bool)):
        raise ValueError("Input must be boolean array")

    if len(bool_array) == 0:
        return np.array([])

    nonzero = np.where(bool_array)[0]  # Find non-zeros
    endzero = np.where(np.diff(nonzero) > 1)[0]  # Find end of non-zero groups
    seq = np.zeros((len(endzero) + 1, 2), dtype=int)  # Initializing array (+1 because last sequence is not calculated))
    seq[:, 1] = nonzero[np.append(endzero, -1)]  # End
    seq[:, 0] = nonzero[np.insert(endzero, 0, -1) + 1]  # Start
    seq = seq[seq[:, 1] - seq[:, 0] != 0]
    return seq
