import pandas as pd
import numpy as np
import scipy.constants
from mobgap.Reorientation.CorrectSensorOrientationDynamic import CorrectSensorOrientationDynamic
from mobgap.gsd import GsdIluz
from mobgap.Reorientation.filteringsignals_100Hz import filtering_signals_100hz
from mobgap.icd._hklee_algo_improved import groupfind
from mobgap.data_transform import (
    SavgolFilter,
    chain_transformers,
)


def CorrectOrientationSensorAxes(data: pd.DataFrame, sampling_rate_hz: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """

    Updates the orientation of the IMU data based on the orientation of the sensor.
    Parameters
    ----------
    data
        Accelerometer and gyroscope data. The data should be in m/s2 and deg/s.
    sampling_rate_hz
        Sampling rate of the data in Hz.

    Returns
    -------
    corIMUdata
        Corrected IMU data.
    corIMUdataSequence
        Sequence of the corrected IMU data including start and stop of each walking bout.

    Notes
    -----
    Deviations from the MATLAB implementation:
    - th constant was amended to 0.65 * 9.81 to convert it to m/s^2 from g-units
    - Within the loop MATLAB calculated the 'start' and 'end' samples by multiplying with fs.
    This is not necessary, since 'start' and 'end' are already in samples.
    - Simple rotation applied in areas of the signal without a walking bout would only occur if more than
    2 walking bouts were present. Now this is applied to the whole signal even if walking bouts are less than 2.
    """

    acc = data.loc[:, ["acc_x", "acc_y", "acc_z"]]

    corIMUdata = data
    corIMUdataSequence = pd.DataFrame(columns=['Start', 'End'])

    # Threshold to test alignment of av with gravity.
    # In matlab it was 0.65 and used with g-units.
    # In the present implementation it is converted to m/s^2
    th = 0.65 * scipy.constants.g

    # parameter for smoothing filter
    n_sgfilt = 9041
    accx = acc.loc[:, ["acc_x"]].values

    # Condition to support the minimal signal length required for the filter parameter
    # low pass filtering of vertical acc
    if n_sgfilt < len(accx):
        av_filt = filtering_signals_100hz(accx, 'low', 0.1, sampling_rate=sampling_rate_hz)

        savgol_win_size_samples = n_sgfilt

        savgol = SavgolFilter(
            window_length_s=savgol_win_size_samples / sampling_rate_hz,
            polyorder_rel=1 / savgol_win_size_samples,
        )

        filter_chain = [("savgol", savgol)]

        av_filt1 = chain_transformers(av_filt, filter_chain, sampling_rate_hz=sampling_rate_hz)

        gs = GsdIluz().detect(acc, sampling_rate_hz=sampling_rate_hz).gs_list_

        gsLabel = np.zeros(len(acc.iloc[:, 0]))
        n = max(gs.shape)
        k = 0

        GS = pd.DataFrame(columns=['Start', 'Stop'])

        if n > 2:
            for i in gs.index:
                GS.loc[i, 'Start'] = gs.loc[i, 'start']
                GS.loc[i, 'Stop'] = gs.loc[i, 'end']

                # start and end are already in samples so no need to multiply with fs as done in MATLAB
                l1 = gs.loc[i, 'start']
                l2 = gs.loc[i, 'end']

                gsLabel[l1:l2] = 1

                avm = np.mean(av_filt1[l1:l2])

                if avm >= th:
                    corIMUdata.iloc[l1:l2, 0:3] = acc.iloc[l1:l2, :]
                elif avm <= -th:
                    corIMUdata.iloc[l1:l2, 0] = -acc.iloc[l1:l2, 0]
                    corIMUdataSequence.loc[k, ['Start', 'End']] = [gs.loc[i, 'start'], gs.loc[i, 'end']]
                    k = k + 1
                else:
                    i = int(i)
                    corIMUdata.iloc[l1:l2, :] = CorrectSensorOrientationDynamic(data.iloc[l1:l2, :], sampling_rate_hz)
                    corIMUdataSequence.loc[k, ['Start', 'End']] = [gs.loc[i, 'start'], gs.loc[i, 'end']]
                    k = k + 1

        ind_noGS = groupfind(gsLabel == 0)

        for i in range(len(ind_noGS[:, 0])):
            l1 = ind_noGS[i, 0]
            l2 = ind_noGS[i, 1]
            avm = np.mean(av_filt1[l1:l2])

            if avm >= th:
                corIMUdata.iloc[l1:l2, 0:3] = acc.iloc[l1:l2, :]
            elif avm <= -th:
                corIMUdata.iloc[l1:l2, 0] = -acc.iloc[l1:l2, 0]

    return corIMUdata, corIMUdataSequence
