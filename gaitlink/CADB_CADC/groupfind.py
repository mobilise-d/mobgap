import numpy as np

def groupfind(L):
    temp = np.where(L)[0]  # Find the indices of nonzero values
    idx = np.where(np.diff(temp) > 1)[0]  # Find where groups end
    ind = np.zeros((len(idx) + 1, 2), dtype=int)  # Initialize the indices array

    ind[:, 1] = temp[np.append(idx, -1)]  # End
    ind[:, 0] = temp[np.insert(idx, 0, -1) + 1]  # Start

    return ind