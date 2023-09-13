import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as np_sliding_window_view


def sliding_window_view(data: np.ndarray, window_size_samples: int, overlap_samples: int):
    return np_sliding_window_view(data, window_shape=(window_size_samples,), axis=0)[
        :: (window_size_samples - overlap_samples)
    ]
