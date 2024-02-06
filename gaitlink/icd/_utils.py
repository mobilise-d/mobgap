from typing import Literal

import numpy as np


def find_zero_crossings(
    signal: np.ndarray,
    mode: Literal["positive_to_negative", "negative_to_positive", "both"] = "both",
    refine: bool = True,
) -> np.ndarray:
    """
    Find zero crossings in a signal.

    Note, that the return values are floating point indices, as we use linear interpolation to refine the position of
    the zero crossing.

    Parameters
    ----------
    signal
        A numpy array of the signal values.
    mode
        A string specifying the type of zero crossings to detect.
        Can be 'positive_to_negative', 'negative_to_positive', or 'both'.
        The default is 'both'.
    refine
        A boolean specifying whether to refine the position of the zero crossing by linear interpolation.
        The default is True.
        If True, the returned indices will be floating point indices.

    Returns
    -------
    np.ndarray
        A numpy array containing the indices where zero crossings occur.

    Raises
    ------
    ValueError
        If the mode is not one of the specified options.

    Examples
    --------
    >>> signal = np.array([1, -1, 1, -1, 1])
    >>> find_zero_crossings(signal, mode="both")
    array([0, 1, 2, 3])

    Notes
    -----
    Under the hood we use :func:`numpy.sign` to find the indices where the sign of the signal changes.
    This function returns `-1 if x < 0, 0 if x==0, 1 if x > 0`.
    We change the output of the function to be `1 if x >= 0, -1 if x < 0`.
    This should handle cases where the value of the data becomes 0 for a couple of cases and then changes sign again.
    Note, that this might result in some unexpected behaviour, when the signal comes from negative values and then
    becomes 0 for a couple of samples and then negative again.
    This will be treated as two 0 crossings.
    However, if the same happens with positive values, no 0 crossing will be detected.
    With "real" data, this should not be a problem, but it is important to keep in mind.

    """
    sign = np.sign(signal)
    # We change the cases where the signal is 0 to be considered as positive.
    sign[sign == 0] = 1

    # Find indices where sign changes (this is the index before the 0 crossing)
    crossings = np.where(np.abs(np.diff(sign)))[0]

    if mode == "positive_to_negative":
        crossings = crossings[signal[crossings] >= 0]
    elif mode == "negative_to_positive":
        crossings = crossings[signal[crossings] < 0]
    elif mode == "both":
        pass
    else:
        raise ValueError("Invalid mode. Choose 'positive_to_negative', 'negative_to_positive', or 'both'.")

    if not refine:
        return crossings
    # Refine the position of the 0 crossing by linear interpolation and identify the real floating point index
    # of the 0 crossing.
    # Basically, we assume the real 0 crossing to be the index and the index + 1.
    # We compare the "absolute" value of the value at index and index + 1, to figure out how close to index the real
    # zero crossing is.
    refined_crossings = crossings + np.abs(signal[crossings] / (signal[crossings] - signal[crossings + 1]))

    return refined_crossings
