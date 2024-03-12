import pandas as pd


def extract_ref_data(datapoint):
    """
    Extracts reference data from a given datapoint.

    This function extracts the IMU data from the "LowerBack" of the datapoint, 
    and the reference walking bouts and initial contacts from the reference parameters of the datapoint.
    It then iterates over the length of the reference walking bouts, and for each bout, it appends the corresponding IMU data, initial contacts, and labels to their respective lists.

    Parameters:
    ----------
    datapoint : (object)
        The datapoint from which to extract the reference data. 
 
    Returns:
    ----------
    tuple: A tuple containing three lists:
           - data_list: A list of pandas DataFrames, each containing the IMU data for a walking bout.
           - ic_list: A list of pandas DataFrames, each containing the initial contacts for a walking bout, zero-indexed relative to the start of the walking bout.
           - label_list: A list of pandas DataFrames, each containing the labels for a walking bout.
    """

    imu_data = datapoint.data["LowerBack"]
    ref_walking_bouts = datapoint.reference_parameters_.wb_list
    ref_ics = datapoint.reference_parameters_.ic_list

    data_list = []
    ic_list = []
    label_list = []

    for gs in range(len(ref_walking_bouts)):
        gs_start = ref_walking_bouts.iloc[gs].start
        gs_end = ref_walking_bouts.iloc[gs].end
        
        data_list.append(imu_data.iloc[gs_start : gs_end].reset_index(drop = True))
        ic_list.append(ref_ics.loc[ref_ics.index.get_level_values('wb_id') == gs + 1, ['ic']].reset_index(drop = True) - gs_start) 
        label_list.append(ref_ics.loc[ref_ics.index.get_level_values('wb_id') == gs + 1, ['lr_label']].reset_index(drop = True))
    
    return data_list, ic_list, label_list