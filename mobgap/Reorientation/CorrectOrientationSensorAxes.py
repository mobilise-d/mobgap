import pandas as pd
import numpy as np
from mobgap.Reorientation.CorrectSensorOrientationDynamic import CorrectSensorOrientationDynamic
from gaitlink.gsd import GsdIluz
from gaitlink.Reorientation.filteringsignals_100Hz import filtering_signals_100hz
from gaitlink.icd._hklee_algo_improved import groupfind
from gaitlink.data_transform import (
    SavgolFilter,
    chain_transformers,
)

def CorrectOrientationSensorAxes(data: pd.DataFrame, sampling_rate_hz: float) -> pd.DataFrame:
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

    """

    Acc = data.iloc[:, 0:3]
    Gyr = data.iloc[:, 3:6]


    corIMUdata = data
    corIMUdataSequence = pd.DataFrame(columns=['Start', 'End'])
    gs = []

    th = 0.65 # threshold to test alignment of av with gravity (+/-1g)
    N_sgfilt = 9041 # parameter Savitzky - Golay smoothing filter

    Accx = Acc.iloc[:, 0].values

    if N_sgfilt < len(Accx): #condition to support the minimal signal length required for the filter parameter
                             #low pass filtering of vertical acc (supposed to be recorded on right channel/IMU data matrix)

        av_filt = filtering_signals_100hz(Acc.iloc[:, 0], 'low', 0.1, sampling_rate=sampling_rate_hz)

        savgol_win_size_samples = N_sgfilt

        savgol = SavgolFilter(
            window_length_s=savgol_win_size_samples / sampling_rate_hz,
            polyorder_rel=1 / savgol_win_size_samples,
        )

        filter = [("savgol", savgol)]

        av_filt1 = chain_transformers(av_filt, filter, sampling_rate_hz=sampling_rate_hz)

        gs = GsdIluz().detect(Acc, sampling_rate_hz=sampling_rate_hz).gs_list_

        gsLabel = np.zeros(len(Acc.iloc[:, 0]))
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

                gsLabel[l1 : l2] = 1

                avm = np.mean(av_filt1[l1 : l2])
                test = 1

                if avm >= th:
                    corIMUdata.iloc[l1:l2, 0:3] = Acc.iloc[l1:l2, :]
                elif avm <= -th:
                    corIMUdata.iloc[l1:l2, 0] = -Acc.iloc[l1:l2, 0]
                    corIMUdataSequence.loc[k, ['Start', 'End']] = [gs.loc[i, 'start'], gs.loc[i, 'end']]
                    k = k + 1

                else:
                    i = int(i)
                    corIMUdata.iloc[l1:l2, :] = CorrectSensorOrientationDynamic(data.iloc[l1:l2, :], sampling_rate_hz)
                    corIMUdataSequence.loc[k, ['Start', 'End']] = [gs.loc[i, 'start'], gs.loc[i, 'end']]
                    k = k + 1

            ind_noGS = groupfind(gsLabel == 0)

            for i in range(1, len(ind_noGS[:, 0])):
                l1 = ind_noGS[i, 0]
                l2 = ind_noGS[i, 1]
                avm = np.mean(av_filt1[l1 : l2])

                if avm >= th:
                    corIMUdata.iloc[l1:l2, 0:3] = Acc.iloc[l1:l2, :]
                elif avm <= -th:
                    corIMUdata.iloc[l1:l2, 0] = -Acc.iloc[l1:l2, 0]

    return corIMUdata, corIMUdataSequence