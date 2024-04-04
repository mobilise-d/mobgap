import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, correlate
from sklearn.decomposition import PCA
from numpy.linalg import norm
from gaitmap.trajectory_reconstruction import MadgwickAHRS
from mobgap.Reorientation.utils import acceleration
from mobgap.data_transform import (
    SavgolFilter,
    chain_transformers,
)

def CorrectSensorOrientationDynamic (data: pd.DataFrame, sampling_rate_hz: float) -> pd.DataFrame:

    '''
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
    -The signal is allready sliced before calling this function. Thus, this script has been simplified
    -All parameters are expressed in the units used in gaitlink, as opposed to matlab.
      Specifically, we use m/s^2 instead of g.

    '''

    Acc = data.iloc[:, 0:3]
    Gyr = data.iloc[:, 3:6]

    # Initiating Madwick with the default initial orientation in the MATLAB implementation
    # (in MATLAB quaternions are represented as xyzw but Madgwick in gaitmap requires wxyz)
    mad = MadgwickAHRS(beta=0.1, initial_orientation=np.array([0.75, 0, 0, 0.8]) / norm(np.array([0.75, 0, 0, 0.8])))

    chosenacc = Acc
    chosengyr = Gyr

    mytime = np.arange(len(chosenacc)) / sampling_rate_hz
    quaternion = np.zeros((len(mytime), 4))

    for t in range(len(mytime)):
        data = pd.concat([chosenacc.iloc[[t]], chosengyr.iloc[[t]]], axis=1) # In Mobilise-D data are already in m/s2 and deg/s as required by the gaitlab_map package
        mad = mad.estimate(data, sampling_rate_hz=sampling_rate_hz)
        quaternion[t, :] = mad.orientation_.iloc[1, :]

    # Adjust quaternion as from x, y, z, w to w, x, y, z
    quaternion = quaternion[:, [3, 0, 1, 2]]

    data = pd.concat((chosenacc, chosengyr), axis=1)
    av = acceleration(data, quaternion)

    #Principal component analysis
    pca_acc = PCA()
    pca_acc.fit(chosenacc)
    pcaCoef = pca_acc.components_
    newAcc = chosenacc.values @ pcaCoef.T

    pca_gyr = PCA()
    pca_gyr.fit(chosengyr)
    pcaCoefGyr = pca_gyr.components_
    newGyr = chosengyr.values @ pcaCoefGyr.T

    if np.mean(newAcc[:, 0]) < 0:
        newAcc[:, 0] = -newAcc[:, 0]

    # av_magpca
    av_magpca = av.copy()
    av_magpca[:, 0] = newAcc[:, 0]

    # gyr_magpca
    gyr_magpca = chosengyr.to_numpy().copy()
    gyr_magpca[:, 0] = newGyr[:, 0]        # Yaw

    # Standardization
    sig1 = (av[:, 2] - np.mean(av[:, 2])) / np.std(av[:, 2])
    sig2 = (newAcc[:, 1] - np.mean(newAcc[:, 1])) / np.std(newAcc[:, 1])
    sig3 = (newAcc[:, 2] - np.mean(newAcc[:, 2])) / np.std(newAcc[:, 2])
    sig4 = (av[:, 1] - np.mean(av[:, 1])) / np.std(av[:, 1])

    # Assigning av_pca
    av_pca = av_magpca

    # Assigning gyr_pca
    gyr_pca = gyr_magpca

    # Calculating correlations
    cor1 = np.dot(sig1.T, sig2)
    cor2 = np.dot(sig1.T, sig3)

    if abs(cor1) > abs(cor2):
        if cor1 > 0:
            av_pca[:, 2] = newAcc[:, 1]  # AP
            gyr_pca[:, 2] = newGyr[:, 1]
        else:
            av_pca[:, 2] = -newAcc[:, 1]  # AP
            gyr_pca[:, 2] = newGyr[:, 1]

        if np.dot(sig3.T, sig4) > 0:
            av_pca[:, 1] = newAcc[:, 2]  # ML
            gyr_pca[:, 1] = newGyr[:, 2]
        else:
            av_pca[:, 1] = -newAcc[:, 2]  # ML
            gyr_pca[:, 1] = newGyr[:, 2]
    else:
        if cor2 > 0:
            av_pca[:, 2] = newAcc[:, 2]  # AP
            gyr_pca[:, 2] = newGyr[:, 2]
        else:
            av_pca[:, 2] = -newAcc[:, 2]  # AP
            gyr_pca[:, 2] = newGyr[:, 2]

        if np.dot(sig2.T, sig4) > 0:
            av_pca[:, 1] = newAcc[:, 1]  # ML
            gyr_pca[:, 1] = newGyr[:, 1]
        else:
            av_pca[:, 1] = -newAcc[:, 1]  # ML
            gyr_pca[:, 1] = newGyr[:, 1]


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

    filter = [("savgol", savgol)]

    av_pca_filt[:, 2] = chain_transformers(av_pca[:, 2], filter, sampling_rate_hz=sampling_rate_hz)
    av_pca_filt[:, 1] = chain_transformers(av_pca[:, 1], filter, sampling_rate_hz=sampling_rate_hz)
    av_pca_filt[:, 0] = chain_transformers(av_pca[:, 0], filter, sampling_rate_hz=sampling_rate_hz)

    # Standardization
    sig5 = (av_pca_filt[:, 2] - np.mean(av_pca_filt[:, 2])) / np.std(av_pca_filt[:, 2])
    sig6 = (av_pca_filt[:, 1] - np.mean(av_pca_filt[:, 1])) / np.std(av_pca_filt[:, 1])
    sig7 = (av_pca_filt[:, 0] - np.mean(av_pca_filt[:, 0])) / np.std(av_pca_filt[:, 0])

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
    return IMU_corrected