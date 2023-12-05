import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from shin_algo_improved import shin_algo_improved
from hklee_algo_improved import hklee_algo_improved
from gaitlink.data import LabExampleDataset
from gaitlink.data import LabExampleDataset
from gaitlink.data._mobilised_matlab_loader import GenericMobilisedDataset


#data=x chooses only the vertical acceleration, data=all chooses all axes
def InitialContactDetection(Accelerometer, fs, GS, algs, data='x'):

    # Check if 'data' is provided, if not, set it to the default 'all' (all axes of accelerometry)
    if not data or len(data) == 0:
        data = 'all'

    if 'x' in data:
        Accelerometer = Accelerometer[:, 0:1]
    elif 'all' in m:
        Accelerometer = Accelerometer


    if isinstance(GS, pd.DataFrame) and not GS.empty:
        SD_Output = {'Start': [], 'End': [], 'IC': []}
        BN = len(GS['Start'])
        startvec = np.zeros(BN, dtype=int)
        stopvec = np.zeros(BN, dtype=int)

        for i in range(BN):
            startvec[i] = int(np.floor(GS['Start'][i] * fs))
            stopvec[i] = int(np.floor(GS['End'][i] * fs))
            chosenacc = Accelerometer[startvec[i]:stopvec[i], :]

            if 'HKLee_Imp' in algs:  # 1
                IC_HKLee_improved = hklee_algo_improved(chosenacc, fs, 'x')
                SD_Output['IC'].append(IC_HKLee_improved)

            if 'Shin_Imp' in algs:  # 2
                IC_Shin_improved = shin_algo_improved(chosenacc, fs, 'x')
                SD_Output['IC'].append(IC_Shin_improved)

            SD_Output['Start'].append(startvec[i])
            SD_Output['End'].append(stopvec[i])

    else:
        print("GS is empty.")
        SD_Output = {}

    return SD_Output