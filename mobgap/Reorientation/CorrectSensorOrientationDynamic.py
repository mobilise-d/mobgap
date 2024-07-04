import pandas as pd
import numpy as np
from scipy.signal import correlate
from sklearn.decomposition import PCA
from gaitmap.trajectory_reconstruction import MadgwickAHRS
from mobgap.Reorientation.utils import acceleration
from mobgap.data_transform import (
    SavgolFilter,
    chain_transformers,
)


def CorrectSensorOrientationDynamic(data: pd.DataFrame, sampling_rate_hz: float) -> pd.DataFrame:

    """
    Corrects the orientation of the IMU data based on the orientation of the sensor.

    Parameters
    ----------
    data
        Accelerometer and gyroscope data. The data should be in m/s2 and deg/s.
    sampling_rate_hz
        Sampling rate of the data in Hz.

    Returns
    -------
    IMU_corrected
        Corrected IMU data.

    Notes
    -----
    Points of deviation from the MATLAB implementation:
    - The Madgwick filter is applied to the entire data array (walking bout) at once, as opposed to row by row in the
    original implementation.
    - The signal is already sliced before calling the present function (CorrectSensorOrientationDynamic).
    Thus, this script has been simplified by removing a loop and a variable which were not needed.
    - In MATLAB quaternions are represented as xyzw but python requires wxyz. Thus, the quaternions are adjusted.
    - Parameters are expressed in the units used in mobgap,specifically, we use m/s^2 instead of g.
    """

    acc = data.loc[:, ["acc_x", "acc_y", "acc_z"]]
    gyr = data.loc[:, ["gyr_x", "gyr_y", "gyr_z"]]

    # Madwick is called with the initial orientation specified inside the Matlab Madgwick function used in the Matlab
    # implementation (UpdateIMUslt2). This choice was made since the first "initial orientation"
    # specified in the Matlab implementation is overwritten in the UpdateIMUslt2 function.
    
    mad = MadgwickAHRS(beta=0.1, initial_orientation=np.array([-0.0863212587840360, -0.639984676042049, 
                                                               0, 0.763523578361070]))

    chosenacc = acc.copy()
    chosengyr = gyr.copy()

    # applying the madgwick algo to the entire data array
    data = pd.concat((chosenacc, chosengyr), axis=1)
    mad = mad.estimate(data, sampling_rate_hz=sampling_rate_hz)
    # Slicing quaternion to make it identical in length to the data
    # this removes the first instance which is the initial orientation
    quaternion = mad.orientation_.iloc[1:, :]

    quaternion = quaternion.to_numpy()

    # Adjust quaternion from x, y, z, w to w, x, y, z
    quaternion = quaternion[:, [3, 0, 1, 2]]

    av = acceleration(data, quaternion)

    # Principal component analysis
    pca_acc = PCA()
    newacc = pca_acc.fit_transform(chosenacc)

    pca_gyr = PCA()
    newgyr = pca_gyr.fit_transform(chosengyr)

    if np.mean(newacc[:, 0]) < 0:
        newacc[:, 0] = -newacc[:, 0]

    # av_magpca
    av_magpca = av.copy()
    av_magpca[:, 0] = newacc[:, 0]

    # gyr_magpca
    gyr_magpca = chosengyr.to_numpy().copy()
    gyr_magpca[:, 0] = newgyr[:, 0]        # Yaw

    # Standardisation vectorised
    av_standardized = (av - np.mean(av, axis=0)) / np.std(av, axis=0)
    newacc_standardized = (newacc - np.mean(newacc, axis=0)) / np.std(newacc, axis=0)

    sig1 = av_standardized[:, 2]
    sig2 = newacc_standardized[:, 1]
    sig3 = newacc_standardized[:, 2]
    sig4 = av_standardized[:, 1]

    # Assigning av_pca and gyr_pca
    av_pca = av_magpca.copy()
    gyr_pca = gyr_magpca.copy()

    # Calculating correlations
    cor1 = np.dot(sig1.T, sig2)
    cor2 = np.dot(sig1.T, sig3)

    if abs(cor1) > abs(cor2):
        if cor1 > 0:
            av_pca[:, 2] = newacc[:, 1]  # AP
            gyr_pca[:, 2] = newgyr[:, 1]
        else:
            av_pca[:, 2] = -newacc[:, 1]  # AP
            gyr_pca[:, 2] = newgyr[:, 1]

        if np.dot(sig3.T, sig4) > 0:
            av_pca[:, 1] = newacc[:, 2]  # ML
            gyr_pca[:, 1] = newgyr[:, 2]
        else:
            av_pca[:, 1] = -newacc[:, 2]  # ML
            gyr_pca[:, 1] = newgyr[:, 2]
    else:
        if cor2 > 0:
            av_pca[:, 2] = newacc[:, 2]  # AP
            gyr_pca[:, 2] = newgyr[:, 2]
        else:
            av_pca[:, 2] = -newacc[:, 2]  # AP
            gyr_pca[:, 2] = newgyr[:, 2]

        if np.dot(sig2.T, sig4) > 0:
            av_pca[:, 1] = newacc[:, 1]  # ML
            gyr_pca[:, 1] = newgyr[:, 1]
        else:
            av_pca[:, 1] = -newacc[:, 1]  # ML
            gyr_pca[:, 1] = newgyr[:, 1]

    # Refinement to provide the IMU axes as standard data matrix
    av_pca_final = av_pca.copy()
    gyr_pca_final = gyr_pca.copy()

    #  Savitzky-Golay filter
    av_pca_filt = np.zeros_like(av_pca)

    savgol_win_size_samples = 11
    savgol = SavgolFilter(
        window_length_s=savgol_win_size_samples / sampling_rate_hz,
        polyorder_rel=0 / savgol_win_size_samples,
    )

    filter_chain = [("savgol", savgol)]

    # Filtering vectorised
    av_pca_filt = np.zeros_like(av_pca)

    for i in range(av_pca.shape[1]):
        av_pca_filt[:, i] = chain_transformers(av_pca[:, i], filter_chain, sampling_rate_hz=sampling_rate_hz)

    # Standardization vectorised
    av_pca_filt_standardized = (av_pca_filt - np.mean(av_pca_filt, axis=0)) / np.std(av_pca_filt, axis=0)

    sig5 = av_pca_filt_standardized[:, 2]
    sig6 = av_pca_filt_standardized[:, 1]
    sig7 = av_pca_filt_standardized[:, 0]

    # Cross-correlation
    r1 = correlate(sig7, sig5)
    r2 = correlate(sig7, sig6)

    if np.max(r1) > np.max(r2):
        av_pca_final[:, 2] = av_pca[:, 2]
        av_pca_final[:, 1] = av_pca[:, 1]

        gyr_pca_final[:, 2] = gyr_pca[:, 2]
        gyr_pca_final[:, 1] = gyr_pca[:, 1]

    else:
        av_pca_final[:, 2] = av_pca[:, 1]
        av_pca_final[:, 1] = av_pca[:, 2]

        gyr_pca_final[:, 2] = gyr_pca[:, 1]
        gyr_pca_final[:, 1] = gyr_pca[:, 2]

    IMU_corrected = np.concatenate((av_pca_final, gyr_pca_final), axis=1)
    IMU_corrected = pd.DataFrame(IMU_corrected)

    return IMU_corrected
