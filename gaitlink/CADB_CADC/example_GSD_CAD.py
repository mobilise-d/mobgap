#TODO: I had to put all files in this folder to call functions same for _filter

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from CADENCE import CADENCE
from shin_algo_improved import shin_algo_improved as Shin_Imp
from hklee_algo_improved import hklee_algo_improved as HKLee_Imp
from gaitlink.data import LabExampleDataset
try:
    from gaitlink.data import LabExampleDataset
except ModuleNotFoundError:
    print("The example_data module is not found, but the program will continue.")
try:
    from gaitlink.data._mobilised_matlab_loader import GenericMobilisedDataset
except ModuleNotFoundError:
    print("The _mobilised_matlab_loader module is not found, but the program will continue.")


#for McR
example_data = LabExampleDataset(reference_system = "INDIP")
ha_example_data = example_data.get_subset(cohort="HA")
single_test = ha_example_data.get_subset(participant_id="002", test="Test11", trial="Trial1")
imu_data = single_test.data["LowerBack"]
imu_acc = imu_data.filter(like="acc")
imu_gyr = imu_data.filter(like="gyr")
fs = single_test.sampling_rate_hz

#manual import acc adn gyr ssame aas MATLAB
#fs = 100
#csv_file = "Accelerometer.csv"
#imu_acc = []
#with open(csv_file, 'r', encoding='utf-8-sig') as file:
#    reader = csv.reader(file)
#    for row in reader:
#        # Convert each element in the row to a float and append the entire row
#        acc = [float(value) for value in row]
#        imu_acc.append(acc)

#csv_file = "Gyroscope.csv"
#imu_gyr = []
#with open(csv_file, 'r', encoding='utf-8-sig') as file:
#    reader = csv.reader(file)
#    for row in reader:
#        gyr = [float(value) for value in row]
#        imu_gyr.append(gyr)



DATA = np.concatenate((imu_acc, imu_gyr), axis=1)

# Gait Sequence Detection algorithm with GSD function
# plot_results = True
# GS = GSD_LowBackAcc(imu_acc, fs, plot_results)

# Gait Sequence Detection from example file
# example_data_with_reference = LabExampleDataset(reference_system="INDIP")
# single_trial_with_reference = example_data_with_reference.get_subset(
#    cohort="HA", participant_id="001", test="Test11", trial="Trial1"
# )



GS = single_test.reference_parameters_['wb']
GS = [{"Start": r["Start"], "End": r["End"]} for r in GS]
GS = pd.DataFrame.from_records(GS)
#print(GS)



# GS from example file (csv) until I resolve the issue importing the GS from the example file
# file_path = "GS.csv"
# df = pd.read_csv(file_path)
# GS = {
#     'Start': df['Start'].to_numpy(),
#     'End': df['End'].to_numpy(),
#     'fs': df['fs'].to_numpy()
# }

#Cadence detection algorithm
algs = ['Shin_Imp', 'HKLee_Imp']
if len(GS) > 1:
    CAD = CADENCE(DATA, fs, GS, 'HKLee_Imp')
    GS = CAD

else:
    print('Error: the Gait sequence (GS) input is empty')

# Optional: save Gait Sequence
# np.save('GS.npy', GS)
