import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, correlate
from sklearn.decomposition import PCA
from numpy.linalg import norm
from gaitmap.trajectory_reconstruction import MadgwickAHRS
from gaitlink.Reorientation.utils import acceleration


def CorrectSensorOrientationDynamic (data: pd.DataFrame, sampling_rate_hz: float, DS: np.ndarray) -> pd.DataFrame:

    Acc = data.iloc[:, 0:3]
    Gyr = data.iloc[:, 3:6]

    NDS = len([x.Start for x in DS])
    startvec = np.zeros(NDS)
    stopvec = np.zeros(NDS)

    # Initiating Madwick with the default initial orientation in the MATLAB implementation
    mad = MadgwickAHRS(beta=0.1, initial_orientation=np.array([0.8, 0.75, 0, 0]) / norm(np.array([0.8, 0.75, 0, 0])))

    for i in range(NDS):

        try:
            startvec[i] = int(np.floor(DS[i]['Start'] * sampling_rate_hz))
            if startvec[i] < 1:
                startvec[i] = 1

            stopvec[i] = int(np.floor(DS[i]['End'] * sampling_rate_hz))
            if stopvec[i] > len(Acc):
                stopvec[i] = len(Acc)
        except IndexError:
            # Handle index out of range errors
            pass

        chosenacc = Acc[startvec[i]:stopvec[i], :]
        chosengyr = Gyr[startvec[i]:stopvec[i], :]

        mytime = np.arange(len(chosenacc)) / sampling_rate_hz
        quaternion = np.zeros((len(mytime), 4))

        for t in range(len(mytime)):
            data = (chosenacc[t, :], chosengyr[t, :]) # In Mobilise-D data are already in m/s2 and deg/s as required by the gaitlab_map package
            mad = mad.estimate(data, sampling_rate_hz=sampling_rate_hz)
            quaternion[t, :] = mad.orientation_

        data = (chosenacc, chosengyr)
        av = acceleration(data, quaternion)

        #Principal component analysis
        pca_acc = PCA()
        pca_acc.fit(chosenacc)
        pcaCoef = pca_acc.components_
        newAcc = np.dot(chosenacc, pcaCoef.T)

        pca_gyr = PCA()
        pca_gyr.fit(chosengyr)
        pcaCoefGyr = pca_gyr.components_
        newGyr = np.dot(chosengyr, pcaCoefGyr.T)

        if np.mean(newAcc[:, 0]) < 0:
            newAcc[:, 0] = -newAcc[:, 0]

        # av_magpca
        av_magpca = av.copy()
        av_magpca[:, 0] = newAcc[:, 0]

        # gyr_magpca
        gyr_magpca = chosengyr.copy()
        gyr_magpca[:, 0] = newGyr[:, 0]  # Yaw

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

        # Applying Savitzky-Golay filter
        av_pca_filt = np.zeros_like(av_pca)
        av_pca_filt[:, 2] = savgol_filter(av_pca[:, 2], 0, 11)
        av_pca_filt[:, 1] = savgol_filter(av_pca[:, 1], 0, 11)
        av_pca_filt[:, 0] = savgol_filter(av_pca[:, 0], 0, 11)

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
