import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from CADENCE import CADENCE
from shin_algo_improved import shin_algo_improved as Shin_Imp
from hklee_algo_improved import hklee_algo_improved as HKLee_Imp
from ICD_CAD import InitialContactDetection
from gaitlink.data import LabExampleDataset
from gaitlink.data import LabExampleDataset
from gaitlink.data._mobilised_matlab_loader import GenericMobilisedDataset

#Load data
example_data = LabExampleDataset(reference_system = "INDIP")
ha_example_data = example_data.get_subset(cohort="HA")
single_test = ha_example_data.get_subset(participant_id="002", test="Test11", trial="Trial1")
imu_data = single_test.data["LowerBack"]
imu_acc = imu_data.filter(like="acc")
imu_gyr = imu_data.filter(like="gyr")
fs = single_test.sampling_rate_hz

GS = single_test.reference_parameters_['wb']
GS = [{"Start": r["Start"], "End": r["End"]} for r in GS]
GS = pd.DataFrame.from_records(GS)

DATA = np.concatenate((imu_acc, imu_gyr), axis=1)

Accelerometer = DATA

ICD = InitialContactDetection(Accelerometer, fs, GS, ['Shin_Imp'], 'x')