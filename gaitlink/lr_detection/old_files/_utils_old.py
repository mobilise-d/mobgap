import numpy as np
from scipy.signal import butter, filtfilt

def _butter_bandpass_filter(signal: np.ndarray,
                            lower_bound: float,
                            upper_bound: float,
                            sampling_rate_hz: float,
                            order = 4) -> np.ndarray:
    """
    This utility function creates and applies a Butterworth lowpass filter.
    """
    nyq = 0.5 * sampling_rate_hz
    normal_lower_bound = lower_bound / nyq
    normal_upper_bound = upper_bound / nyq
    b, a = butter(order, [normal_lower_bound, normal_upper_bound], btype = "bandpass", analog = False)
    y = filtfilt(b, a, signal)
    return y

def _butter_lowpass_filter(signal: np.ndarray,
                           cutoff: float,
                           sampling_rate_hz: float,
                           order = 4) -> np.ndarray:
    """Create and apply butterworth lowpass filter."""
    nyq = 0.5 * sampling_rate_hz
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype = "low", analog = False)
    y = filtfilt(b, a, signal)
    return y


def find_extrema_in_radius(data: np.ndarray,
                           indices: np.ndarray,
                           radius_left: int,
                           radius_right: int,
                           extrema_type: "min"):
    """Returns the index of the global extrema of data in the given radius around each index in indices.

    Parameters
    ----------
    data : 1D array
        Data used to find the extrema
    indices : 1D array of ints
        Around each index the extremum is searched in the region defined by radius
    radius_left
        The number of samples to the left for the search.
    radius_right
        The number of samples to the right for the search.
    extrema_type
        If the minima or maxima of the data are searched.

    Returns
    -------
    list_of_extrema_indices
        Array of the position of each identified extremum

    """
    extrema_funcs = {"min": np.nanargmin, "max": np.nanargmax}
    if extrema_type not in extrema_funcs:
        raise ValueError("`extrema_type` must be one of {}, not {}".format(list(extrema_funcs.keys()), extrema_type))
    extrema_func = extrema_funcs[extrema_type]
    # Search region is twice the radius centered around each index
    d = radius_left + radius_right + 1
    start_padding = 0

    data = data.astype(float)
    # apply zero padding to the right in case the highest search index + radius_right exceeds len(data)
    if len(data) - np.max(indices) <= radius_right:
        data = np.pad(data, (0, radius_right), mode="constant", constant_values=np.nan)
    # apply zero padding to the left in case the smallest search index - radius is < 0
    if np.min(indices) < radius_left:
        start_padding = radius_left
        data = np.pad(data, (start_padding, 0), mode="constant", constant_values=np.nan)
    strides = sliding_window_view(data, window_length=d, overlap=d - 1)
    # select all windows around indices
    windows = strides[indices.astype(int) - radius_left + start_padding, :]
    return extrema_func(windows, axis=1) + indices - radius_left

def sliding_window_view(arr: np.ndarray, window_length: int, overlap: int, nan_padding: bool = False) -> np.ndarray:
    """Create a sliding window view of an input array with given window length and overlap.

    .. warning::
        This function will return by default a view onto your input array, modifying values in your result will directly
        affect your input data which might lead to unexpected behaviour! If padding is disabled (default) last window
        fraction of input may not be returned! However, if nan_padding is enabled, this will always return a copy
        instead of a view of your input data, independent if padding was actually performed or not!

    Parameters
    ----------
    arr : array with shape (n,) or (n, m)
        array on which sliding window action should be performed. Windowing
        will always be performed along axis 0.

    window_length : int
        length of desired window (must be smaller than array length n)

    overlap : int
        length of desired overlap (must be smaller than window_length)

    nan_padding: bool
        select if last window should be nan-padded or discarded if it not fits with input array length. If nan-padding
        is enabled the return array will always be a copy of the input array independent if padding was actually
        performed or not!

    Returns
    -------
    windowed view (or copy for nan_padding) of input array as specified, last window might be nan padded if necessary to
    match window size

    Examples
    --------
    >>> data = np.arange(0,10)
    >>> windowed_view = sliding_window_view(arr = data, window_length = 5, overlap = 3, nan_padding = True)
    >>> windowed_view
    np.array([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6], [4, 5, 6, 7, 8], [6, 7, 8, 9, np.nan]])

    """
    if overlap >= window_length:
        raise ValueError("Invalid Input, overlap must be smaller than window length!")

    if window_length < 2:
        raise ValueError("Invalid Input, window_length must be larger than 1!")

    # calculate length of necessary np.nan-padding to make sure windows and overlaps exactly fits data length
    n_windows = np.ceil((len(arr) - window_length) / (window_length - overlap)).astype(int)
    pad_length = window_length + n_windows * (window_length - overlap) - len(arr)

    # had to handle 1D arrays separately
    if arr.ndim == 1:
        if nan_padding:
            # np.pad always returns a copy of the input array even if pad_length is 0!
            arr = np.pad(arr.astype(float), (0, pad_length), constant_values=np.nan)

        new_shape = (arr.size - window_length + 1, window_length)
    else:
        if nan_padding:
            # np.pad always returns a copy of the input array even if pad_length is 0!
            arr = np.pad(arr.astype(float), [(0, pad_length), (0, 0)], constant_values=np.nan)

        shape = (window_length, arr.shape[-1])
        n = np.array(arr.shape)
        o = n - shape + 1  # output shape
        new_shape = np.concatenate((o, shape), axis=0)

    # apply stride_tricks magic
    new_strides = np.concatenate((arr.strides, arr.strides), axis=0)
    view = np.lib.stride_tricks.as_strided(arr, new_shape, new_strides)[0 :: (window_length - overlap)]

    view = np.squeeze(view)  # get rid of single-dimensional entries from the shape of an array.

    return view
    