import pandas as pd
import numpy as np
import scipy.constants
from typing_extensions import Self
from mobgap.pipeline import iter_gs
from mobgap.Reorientation.CorrectSensorOrientationDynamic import CorrectSensorOrientationDynamic
from mobgap.gsd import GsdIluz
from mobgap.Reorientation._filteringsignals_100Hz import filtering_signals_100hz
from mobgap.icd._hklee_algo_improved import groupfind
from mobgap.data_transform import (
    SavgolFilter,
    chain_transformers,
)


class CorrectOrientationSensorAxes:
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


    corIMUdata: pd.DataFrame
    corIMUdataSequence: pd.DataFrame

    def __init__(self, sampling_rate_hz: float):
        self.sampling_rate_hz = sampling_rate_hz

    def update_orientation(self, data: pd.DataFrame) -> Self:
        self.data = data

        acc = self.data.loc[:, ["acc_x", "acc_y", "acc_z"]]

        self.corIMUdata = self.data.copy()
        self.corIMUdataSequence = pd.DataFrame(columns=['Start', 'End'])

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
            av_filt = filtering_signals_100hz(accx, 'low', 0.1, sampling_rate=self.sampling_rate_hz)

            savgol_win_size_samples = n_sgfilt

            savgol = SavgolFilter(
                window_length_s=savgol_win_size_samples / self.sampling_rate_hz,
                polyorder_rel=1 / savgol_win_size_samples,
            )

            filter_chain = [("savgol", savgol)]

            av_filt1 = chain_transformers(av_filt, filter_chain, sampling_rate_hz=self.sampling_rate_hz)

            # Output of filter chains should be a pd.DataFrame so the iter_gs function can work
            av_filt1 = pd.DataFrame(av_filt1, columns=['acc_x'])

            gs = GsdIluz().detect(acc, sampling_rate_hz=self.sampling_rate_hz).gs_list_

            # Adding a specific index to gs so the iter_gs function can work
            if gs.index.name is None:
                gs['gs_id'] = range(len(gs))
            else:
                if gs.index.name != 'gs_id' and gs.index.name != 'wb_id':
                    gs['gs_id'] = range(len(gs))

            gsLabel = np.zeros(len(acc.loc[:, "acc_x"]))
            n = len(gs)
            k = 0

            if n > 2:
                for GS, data_slice in iter_gs(av_filt1, gs):

                    gsLabel[GS.start:GS.end] = 1

                    avm = np.mean(data_slice)

                    if avm >= th:
                        # TODO: check maybe this is not needed because corIMUdata is allready = data so acc is there
                        self.corIMUdata.iloc[GS.start:GS.end, 0:3] = acc.iloc[GS.start:GS.end, :]
                    elif avm <= -th:
                        self.corIMUdata.iloc[GS.start:GS.end, 0] = -acc.iloc[GS.start:GS.end, 0]
                        self.corIMUdataSequence.loc[k, ['Start', 'End']] = [GS.start, GS.end]
                        k = k + 1
                    else:
                        self.corIMUdata.iloc[GS.start:GS.end, :] = CorrectSensorOrientationDynamic(
                            self.data.iloc[GS.start:GS.end, :], self.sampling_rate_hz)
                        self.corIMUdataSequence.loc[k, ['Start', 'End']] = [GS.start, GS.end]
                        k = k + 1

            ind_noGS = groupfind(gsLabel == 0)

            # ind_noGS should be a pd.DataFrame so the iter_gs function can work
            ind_noGS = pd.DataFrame(ind_noGS, columns=['start', 'end'])

            # Adding a specific index to ind_noGS so the iter_gs function can work
            if ind_noGS.index.name is None:
                ind_noGS['gs_id'] = range(len(ind_noGS))
            else:
                if ind_noGS.index.name != 'gs_id' and ind_noGS.index.name != 'wb_id':
                    ind_noGS['gs_id'] = range(len(ind_noGS))

            for GS, data_slice in iter_gs(av_filt1, ind_noGS):
                avm = np.mean(data_slice)

                if avm >= th:
                    self.corIMUdata.iloc[GS.start:GS.end, 0:3] = acc.iloc[GS.start:GS.end, :]
                elif avm <= -th:
                    self.corIMUdata.iloc[GS.start:GS.end, 0] = -acc.iloc[GS.start:GS.end, 0]

        else:
            return self

        return self
