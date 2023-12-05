import numpy as np
from scipy.interpolate import interp1d

def resampInterp(y, fs_initial, fs_final):
    if len(y) > 0:
        # Resampling using interpolation from fs_initial to fs_final
        recordingTime = len(y)
        x = np.arange(1, recordingTime + 1)
        xq = np.arange(1, min(recordingTime + 1, recordingTime * (fs_initial / fs_final) + 1), fs_initial / fs_final)
        f = interp1d(x, y, kind='linear', axis=0)
        yResamp = f(xq)
        return yResamp
    else:
        raise ValueError("The input signal is empty.")
