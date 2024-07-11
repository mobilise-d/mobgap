import pandas as pd
import numpy as np
from scipy.signal import correlate
from sklearn.decomposition import PCA
from gaitmap.trajectory_reconstruction import MadgwickAHRS
from mobgap.Reorientation._utils import acceleration
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

    # newacc to pd dataframe so named access can be used later
    newacc = pd.DataFrame(newacc, columns=['PC1', 'PC2', 'PC3'])

    pca_gyr = PCA()
    newgyr = pca_gyr.fit_transform(chosengyr)

    # newgyr to pd dataframe so named access can be used later
    newgyr = pd.DataFrame(newgyr, columns=['PC1', 'PC2', 'PC3'])

    if newacc['PC1'].mean() < 0:
        newacc['PC1'] = -newacc['PC1']

    # av_magpca
    av_magpca = av.copy()
    av_magpca.loc[:, 'acc_x'] = newacc.loc[:, 'PC1']    # Vertical acceleration is replaced by the first PCA component,
                                                        # indicating the higher variance (like the vertical acceleration)

    # gyr_magpca
    gyr_magpca = chosengyr.reset_index(drop=True).copy()    # Resetting index (to remove time) to match the newgyr index
    gyr_magpca.loc[:, 'gyr_x'] = newgyr.loc[:, 'PC1']        # Yaw: Vertical gyroscope is replaced by the first PCA component,
                                                            # indicating the higher variance (yaw)

    # Standardisation vectorised
    av_standardized = ((av - av.mean()) / av.std())
    newacc_standardized = ((newacc - newacc.mean()) / newacc.std())

    sig1 = av_standardized.loc[:, 'acc_z']
    sig2 = newacc_standardized.loc[:, 'PC2']
    sig3 = newacc_standardized.loc[:, 'PC3']
    sig4 = av_standardized.loc[:, 'acc_y']

    # Calculating dot products to compare directionality and magnitude of agreement between different axes
    # The PCA components are ordered by variance explained: the first component has the highest variance,
    # followed by the second, and then the third.
    cor1 = np.dot(sig1, sig2)   # 'agreement' between AP and 2nd PCA component
    cor2 = np.dot(sig1, sig3)   # 'agreement' between AP and 3rd PCA component
    cor3 = np.dot(sig3, sig4)   # 'agreement' between 3rd PCA component and ML
    cor4 = np.dot(sig2, sig4)   # 'agreement' between ML and 2nd PCA component

    if abs(cor1) > abs(cor2):   # AP and 2nd PCA component are more 'aligned' than AP and 3rd PCA component
        if cor1 > 0:    # AP and ML have same direction
            av_magpca.loc[:, 'acc_z'] = newacc.loc[:, 'PC2']  # AP is replaced with 2nd component of PCA
            gyr_magpca.loc[:, 'gyr_z'] = newgyr.loc[:, 'PC2']
        else:   # AP and ML have opposite direction
            av_magpca.loc[:, 'acc_z'] = -newacc.loc[:, 'PC2']  # AP is replaced with the neg of the 2nd PCA component
            gyr_magpca.loc[:, 'gyr_z'] = newgyr.loc[:, 'PC2']

        if cor3 > 0:  # AP (PCA) and ML have same direction
            av_magpca.loc[:, 'acc_y'] = newacc.loc[:, 'PC3']  # ML is replaced with the 3rd component of PCA
            gyr_magpca.loc[:, 'gyr_y'] = newgyr.loc[:, 'PC3']
        else:   # AP (PCA) and ML have opposite direction
            av_magpca.loc[:, 'acc_y'] = -newacc.loc[:, 'PC3']  # ML is replaced with the neg of the 3rd PCA component
            gyr_magpca.loc[:, 'gyr_y'] = newgyr.loc[:, 'PC3']
    else:   # AP and AP (following PCA) are more 'aligned' than AP and ML
        if cor2 > 0:    # AP and AP (PCA) have same direction
            av_magpca.loc[:, 'acc_z'] = newacc.loc[:, 'PC3']  # AP is replaced with the 3rd component of PCA
            gyr_magpca.loc[:, 'gyr_z'] = newgyr.loc[:, 'PC3']
        else:   # AP and AP (PCA) have opposite direction
            av_magpca.loc[:, 'acc_z'] = -newacc.loc[:, 'PC3']  # AP is replaced with the neg of the 3rd PCA component
            gyr_magpca.loc[:, 'gyr_z'] = newgyr.loc[:, 'PC3']

        if cor4 > 0:    # ML and ML (PCA) have same direction
            av_magpca.loc[:, 'acc_y'] = newacc.loc[:, 'PC2']  # ML is replaced with 2nd component of PCA
            gyr_magpca.loc[:, 'gyr_y'] = newgyr.loc[:, 'PC2']
        else:   # ML and ML (PCA) have opposite direction
            av_magpca.loc[:, 'acc_y'] = -newacc.loc[:, 'PC2']  # ML is replaced with the neg of the 2nd component of PCA
            gyr_magpca.loc[:, 'gyr_y'] = newgyr.loc[:, 'PC2']
    av_pca_final = av_magpca.copy()
    gyr_pca_final = gyr_magpca.copy()

    #  Savitzky-Golay filter
    savgol_win_size_samples = 11
    savgol = SavgolFilter(
        window_length_s=savgol_win_size_samples / sampling_rate_hz,
        polyorder_rel=0 / savgol_win_size_samples,
    )

    filter_chain = [("savgol", savgol)]

    # Filtering vectorised
    av_pca_filt = np.zeros_like(av_magpca)

    for i in range(av_magpca.shape[1]):
        av_pca_filt[:, i] = chain_transformers(av_magpca.iloc[:, i], filter_chain, sampling_rate_hz=sampling_rate_hz)

    # Standardization vectorised
    av_pca_filt_standardized = (av_pca_filt - np.mean(av_pca_filt, axis=0)) / np.std(av_pca_filt, axis=0)

    sig5 = av_pca_filt_standardized[:, 2]
    sig6 = av_pca_filt_standardized[:, 1]
    sig7 = av_pca_filt_standardized[:, 0]

    # Cross-correlation
    r1 = correlate(sig7, sig5)  # Cross-correlation indicating lag of signals, this uses the filtered signals
    r2 = correlate(sig7, sig6)

    if np.max(r1) > np.max(r2):    # AP and V correlate more strongly than V and ML. Nothing changes
        av_pca_final.loc[:, "acc_z"] = av_magpca.loc[:, "acc_z"]
        av_pca_final.loc[:, "acc_y"] = av_magpca.loc[:, "acc_y"]

        gyr_pca_final.loc[:, "gyr_z"] = gyr_magpca.loc[:, "gyr_z"]
        gyr_pca_final.loc[:, "gyr_y"] = gyr_magpca.loc[:, "gyr_y"]

    else:   # V and ML correlate more strongly than AP and V.
        av_pca_final.loc[:, "acc_z"] = av_magpca.loc[:, "acc_y"]   # AP is replaced with ML
        av_pca_final.loc[:, "acc_y"] = av_magpca.loc[:, "acc_z"]

        gyr_pca_final.loc[:, "gyr_z"] = gyr_magpca.loc[:, "gyr_y"]
        gyr_pca_final.loc[:, "gyr_y"] = gyr_magpca.loc[:, "gyr_z"]

    IMU_corrected = pd.concat([av_pca_final, gyr_pca_final], axis=1)

    return IMU_corrected
